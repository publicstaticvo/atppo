import os
import re
import tqdm
import time
import torch
import random
import argparse
import numpy as np
import torch.distributed as dist
from apex import amp
from apex.optimizers import FusedAdam
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import DownstreamDataset, DataCollatorForDownstream
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup
from downstream_metrics import downstream_metrics
from sklearn.metrics import accuracy_score
from utils import ATConfig, get_rank
from model import DownstreamModel

os.environ["NCCL_DEBUG"] = "WARN"
LABEL_NUM = {'mosi': 1, 'meld': 7, 'snips': 7, 'mosei': 1, 'mintrec': 20, 'iemocap': 6}
KEY_METRIC_INDEX = {'mosi': 5, 'meld': 1, 'mosei': 5, 'mintrec': 1, 'iemocap': 1}
model_has_audio_cls = [r"v1\.3\.[3-9]", r"v2\.3", r"v3", r"v4"]
v2_ckpt = ["v2", "v3.2", "v3.3", "v4"]
SYSTEM = "/mnt/ewwe/yts"
SAMPLE_RATE = 16000
CONFIG = "config.json"

if __name__ == '__main__':
    # 1. arguments and config
    parser = argparse.ArgumentParser()
    parser.add_argument('--accumulate_num', type=int, default=1)
    parser.add_argument('--audio_path', type=str, default="models/wavlm-base-plus")
    parser.add_argument("--audio_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cl_mode", type=str, default="no", choices=['no', 'step', 'epoch'])
    parser.add_argument("--cl_steps", type=int, default=3)
    parser.add_argument("--dont_show", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--last_conv_layer", type=str, default="no", choices=["no", "layer", "group"])
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument("--multi_audio", action="store_true")
    parser.add_argument('--output_file', type=str, default="results.csv")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pretrain_data", type=int, default=960)
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--save_epoch", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--system', type=str, default="")
    parser.add_argument("--task", type=str, choices=['iemocap', 'mosi', 'meld', 'mintrec', 'mosei'])
    parser.add_argument('--text_path', type=str, default="models/roberta-base")
    parser.add_argument('--train_mode', type=str, default="")
    parser.add_argument("--use_turn_ids", action="store_true")
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
    if args.system:
        SYSTEM = args.system
    audio_length = SAMPLE_RATE * args.audio_length
    model_name = args.model.split('/')[-1]
    if get_rank() == 0:
        print(f"Model {model_name} datasize {args.pretrain_data} batchsize {args.batch_size} epochs {args.epochs}"
              f" lr {args.lr:.1e} gradacc {args.accumulate_num} task {args.task} last_conv_layer {args.last_conv_layer}"
              f" cl_mode {args.cl_mode} cl_steps {args.cl_steps} prompt {args.prompt} train_mode {args.train_mode}")
    model_name = model_name.split('-')
    args.audio_path, args.output_file, args.text_path = map(
        lambda x: os.path.join(SYSTEM, x),
        [args.audio_path, args.output_file, args.text_path])
    args.model = os.path.join("/mnt/shared/public/yts/Audio-Text-Pretraining", args.model)
    large = "L" in model_name[0]
    label_num = LABEL_NUM[args.task]
    v2 = any(n in model_name[0] for n in v2_ckpt)
    try:
        config = ATConfig.from_pretrained(args.model, return_kwargs=False)
    except:
        config = ATConfig.from_json_files(os.path.join(args.audio_path, CONFIG), os.path.join(args.text_path, CONFIG))
    config.audio.has_audio_cls = any(re.search(s, model_name[0]) for s in model_has_audio_cls)
    config.audio.use_turn_ids = args.multi_audio or args.use_turn_ids
    config.audio.last_conv_layer = args.last_conv_layer
    config.audio.multi_turn = args.multi_audio
    config.set_length(audio_length, 512)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_path)
    # 2. seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # 3. load model
    turn_embeddings = torch.load(os.path.join(SYSTEM, "models/bert-base/pytorch_model.bin")).pop("bert.embeddings.token_type_embeddings.weight") if "v3" not in model_name[0] else None
    print(f"has_audio_cls {config.audio.has_audio_cls} multi audio {config.audio.multi_turn} v2 {v2}" 
           f"prompt {args.prompt} bert {turn_embeddings is not None} scheduler_type {args.warmup}")
    model = DownstreamModel(args.model, config, label_num, turn_embeddings=turn_embeddings).to(args.device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.lr, bias_correction=False)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale="dynamic")
    # 4. dataset
    c = DataCollatorForDownstream(audio_length, args.task in ["mosi", "mosei"], 180 if args.multi_audio else 90, args.prompt)
    train_data = DownstreamDataset(SYSTEM, args.task, "train", args.train_mode, tokenizer, v2, args.prompt, args.multi_audio)
    if n_gpu > 1:
        model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=[args.local_rank])
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=c,
                                  sampler=DistributedSampler(train_data), pin_memory=True)
    else:
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=c,
                                  sampler=RandomSampler(train_data))
    valid_loader = DataLoader(dataset=DownstreamDataset(SYSTEM, args.task, "valid", args.train_mode, tokenizer, v2, args.prompt, args.multi_audio), batch_size=args.batch_size, collate_fn=c)
    test_loader = DataLoader(dataset=DownstreamDataset(SYSTEM, args.task, "test", args.train_mode, tokenizer, v2, args.prompt, args.multi_audio), batch_size=args.batch_size, collate_fn=c)
    # 5. scheduler
    if args.warmup > 0:
        steps = args.epochs * ((len(train_data) - 1) // args.batch_size // args.accumulate_num // max(1, n_gpu) + 1)
        warmup_steps = int(args.warmup * steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 6.preparation
    experiment_data = [[], [], [], []] if args.task in ["mosi", "mosei"] else [[], [], []]
    early_stop_metric = [-10.0, 0.0, 0.0, 0.0] if args.task in ["mosi", "mosei"] else [-10.0, 0.0, 0.0]
    equal = [False for _ in early_stop_metric]
    best_epoch = 0
    progress = 0
    best_metrics = []
    if args.cl_mode == "step":
        args.cl_steps = args.cl_steps * len(train_loader)
    for epoch in range(args.epochs):
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
            if args.cl_mode == "step":
                progress += 1
        if not args.dont_show and n_gpu <= 1:
            print(f"Epoch {epoch:03d} average loss {torch.mean(torch.tensor(epoch_train_loss)):.4f}")
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
        pred_y, true_y = [], []
        with torch.no_grad():
            time.sleep(1)
            for batch in test_loader:
                batch = {k: (v.to(args.device) if v is not None else None) for k, v in batch.items()}
                logits = model(batch["audio"], batch["text"], batch["aam"], batch["tam"], batch["turn_id"], prompt=batch["prompt"])
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
        if args.cl_mode == "epoch":
            progress += 1
        if epoch - best_epoch == args.patience or (early_stop_metric[-1] == 0.0 and epoch == 2):
            if get_rank() == 0:
                print(f"early stopping at {epoch + 1}")
            break
    if get_rank() == 0:
        params = ['-'.join([model_name[0], model_name[-1]]), str(args.pretrain_data), str(args.batch_size * n_gpu),
                  str(args.epochs), str(args.lr), str(args.accumulate_num), str(args.seed), args.task, args.last_conv_layer,
                  str(args.multi_audio), str(args.warmup), str(v2), str(args.prompt), args.cl_mode, str(args.cl_steps), args.train_mode]
        experiment_data.append(best_metrics)
        with open(args.output_file, "a+") as f:
            f.write("\n" + ",".join(params + [",".join([f"{r}" if isinstance(r, int) else f"{r:.4g}" for r in exp])
                                              for exp in experiment_data]))
