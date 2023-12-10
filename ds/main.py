import os
import sys
import math
import tqdm
import time
import random
import argparse
import numpy as np
from apex import amp
from apex.optimizers import FusedAdam
from modeling_at import ATForSequenceClassification
from torch.nn.parallel import DistributedDataParallel as DDP
from ds_dataloader import DownstreamDataset, DataCollatorForDownstream
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup
from downstream_metrics import downstream_metrics
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from util import *

os.environ["NCCL_DEBUG"] = "WARN"
LABEL_NUM = {'mosi': 1, 'meld': 7, 'snips': 7, 'mosei': 1, 'mintrec': 20, 'iemocap': 6}
KEY_METRIC_INDEX = {'mosi': 5, 'meld': 1, 'mosei': 5, 'mintrec': 1, 'iemocap': 1}
DATA_PATH = "/mnt/ewwe/yts/at/"
MODEL_PATH = "/mnt/ewwe/yts/at/atppo/saved_models"
RESULT_PATH = "/mnt/ewwe/yts/at/atppo/ds/results"
SAMPLE_RATE = 16000
CONFIG = "config.json"

if __name__ == '__main__':
    # 1. arguments and config
    # 默认参数为：iemocap使用multi audio和audio的token type id，数据集使用V2数据集，没有CL、情绪盘等复杂内容。
    parser = argparse.ArgumentParser()
    parser.add_argument('--accumulate_num', type=int, default=1)
    parser.add_argument("--audio_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--min_text_length', type=int, default=2)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output_file', type=str, default="results.csv")
    parser.add_argument("--save_epoch", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--task", type=str, choices=['iemocap', 'mosi', 'meld', 'mintrec', 'mosei'])
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--warmup', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        args.apex_level = 0
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", init_method='env://')
    audio_length = SAMPLE_RATE * args.audio_length
    if get_rank() == 0:
        print(f"Model {args.model} batchsize {args.batch_size} epochs {args.epochs} lr {args.lr:.1e} gradacc {args.accumulate_num} task {args.task} scheduler_type {args.warmup}")
    args.tokenizer = os.path.join(DATA_PATH, args.tokenizer)
    args.output_file = os.path.join(RESULT_PATH, args.output_file)
    args.model = os.path.join(MODEL_PATH, args.model)
    label_num = LABEL_NUM[args.task]
    config = ATConfig.from_pretrained(args.model, return_kwargs=False)
    config.set_length(audio_length, 512)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer)
    # 2. seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # 3. load model
    model = ATForSequenceClassification(args.model, config, label_num).to(args.device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.lr, bias_correction=False)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale="dynamic")
    # 4. dataset
    c = DataCollatorForDownstream(audio_length, args.task in ["mosi", "mosei"], args.min_text_length)
    train_data = DownstreamDataset(DATA_PATH, args.task, "train")
    if n_gpu > 1:
        model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=[args.local_rank])
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=c, sampler=DistributedSampler(train_data), pin_memory=True, num_workers=20)
    else:
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=c, sampler=RandomSampler(train_data), num_workers=20)
    valid_loader = DataLoader(dataset=DownstreamDataset(DATA_PATH, args.task, "valid"), batch_size=args.batch_size, collate_fn=c)
    test_loader = DataLoader(dataset=DownstreamDataset(DATA_PATH, args.task, "test"), batch_size=args.batch_size, collate_fn=c)
    # 5. scheduler
    if args.warmup > 0:
        steps = args.epochs * math.ceil(len(train_data) / args.batch_size / args.accumulate_num / max(1, n_gpu))
        warmup_steps = int(args.warmup * steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 6.preparation
    experiment_data = [[], [], [], []] if args.task in ["mosi", "mosei"] else [[], [], []]
    early_stop_metric = [-10.0, 0.0, 0.0, 0.0] if args.task in ["mosi", "mosei"] else [-10.0, 0.0, 0.0]
    equal = [False for _ in early_stop_metric]
    best_epoch = 0
    best_metrics = []
    for epoch in range(args.epochs):
        # train
        model.train()
        epoch_train_loss = []
        time.sleep(1)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_it = train_loader if args.dont_show else tqdm.tqdm(train_loader)
        for (count, batch) in enumerate(train_it):
            batch = {k: (v.to(args.device) if v is not None else None) for k, v in batch.items()}
            _, loss = model(batch["audio"], batch["text"], batch["aam"], batch["tam"], batch["turn_id"], batch["label"])
            if n_gpu <= 1:
                epoch_train_loss.append(float(loss.detach().cpu()))
                if not args.dont_show:
                    train_it.set_postfix_str(f"loss: {loss:.4f}")
            loss = loss / args.accumulate_num
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if (count + 1) % args.accumulate_num == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        if not args.dont_show and n_gpu <= 1:
            print(f"Epoch {epoch:03d} average loss {torch.mean(torch.tensor(epoch_train_loss)):.4f}")
        # validation
        model.eval()
        epoch_val_loss = []
        pred_y, true_y = [], []
        with torch.no_grad():
            time.sleep(1)
            for batch in valid_loader:
                batch = {k: (v.to(args.device) if v is not None else None) for k, v in batch.items()}
                logits, loss = model(batch["audio"], batch["text"], batch["aam"], batch["tam"], batch["turn_id"], batch["label"])
                if label_num == 1:
                    prediction = logits.view(-1)
                    label_outputs = prediction.cpu().detach().numpy().astype(float)
                else:
                    prediction = torch.argmax(logits, dim=1)
                    label_outputs = prediction.cpu().detach().numpy().astype(int)
                pred_y.extend(label_outputs.tolist())
                true_y.extend(batch["label"].detach().cpu().numpy().tolist())
                epoch_val_loss.append(float(loss.detach().cpu()))
        average_valid_loss = torch.mean(torch.tensor(epoch_val_loss))
        if args.task in ["mosi", "mosei"]:
            m = downstream_metrics(pred_y, true_y, args.task)
            val_acc, val_acc_2 = m["acc_a7"], m["acc_a2_non0"]
            metrics = [-average_valid_loss, val_acc, val_acc_2, val_acc * 5 - average_valid_loss]
        else:
            val_acc = accuracy_score(true_y, pred_y)
            metrics = [-average_valid_loss, val_acc, val_acc * 5 - average_valid_loss]
        for i in range(len(metrics)):
            if metrics[i] >= early_stop_metric[i]:
                equal[i] = (metrics[i] == early_stop_metric[i])
                early_stop_metric[i] = metrics[i]
                best_epoch = epoch
            else:
                equal[i] = False
        if not args.dont_show and get_rank() == 0:
            print(f"Epoch {epoch:03d} average valid loss {average_valid_loss:.4f} valid accuracy {val_acc:.4f} "
                  f"early stop {metrics[-1]:.4f}")
        # test
        pred_y, true_y = [], []
        with torch.no_grad():
            time.sleep(1)
            for batch in test_loader:
                batch = {k: (v.to(args.device) if v is not None else None) for k, v in batch.items()}
                logits = model(batch["audio"], batch["text"], batch["aam"], batch["tam"], batch["turn_id"])
                if label_num == 1:
                    prediction = logits.view(-1)
                    label_outputs = prediction.cpu().detach().numpy().astype(float)
                else:
                    prediction = torch.argmax(logits, dim=1)
                    label_outputs = prediction.cpu().detach().numpy().astype(int)
                pred_y.extend(label_outputs.tolist())
                true_y.extend(batch['label'].detach().cpu().numpy().tolist())
        metric = downstream_metrics(pred_y, true_y, args.task)
        if not args.dont_show and get_rank() == 0:
            print("Test Metric: {}".format(' - '.join(['{}: {:.4f}'.format(k, v) for k, v in metric.items()])))
        for i in range(len(metrics)):
            if early_stop_metric[i] == metrics[i]:
                result = [float(f"{v * 100:.2f}") if "f1" in k or "acc" in k else v for k, v in metric.items()]
                if not experiment_data[i] or not equal[i] or experiment_data[i][KEY_METRIC_INDEX[args.task]] <=\
                        result[KEY_METRIC_INDEX[args.task] - 1]:
                    experiment_data[i] = [epoch + 1] + result
                if not best_metrics or best_metrics[KEY_METRIC_INDEX[args.task]] <= result[KEY_METRIC_INDEX[args.task] - 1]:
                    best_metrics = [epoch + 1] + result
        if epoch - best_epoch == args.patience or (early_stop_metric[-1] == 0.0 and epoch == 2):
            if get_rank() == 0:
                print(f"early stopping at {epoch + 1}")
            break
    if get_rank() == 0:
        params = [args.model, str(args.batch_size * n_gpu), str(args.epochs), str(args.lr), str(args.accumulate_num), args.task, str(args.warmup)]
        experiment_data.append(best_metrics)
        with open(args.output_file, "a+") as f:
            f.write("\n" + ",".join(params + [",".join([f"{r}" if isinstance(r, int) else f"{r:.4g}" for r in exp]) for exp in experiment_data]))
