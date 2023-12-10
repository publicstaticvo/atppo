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
from torch.nn.functional import normalize
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import DownstreamDataset, DataCollatorForDownstream, EpochDataset
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup
from downstream_metrics import downstream_metrics
from sklearn.metrics import accuracy_score
from utils import ATConfig, get_rank
from model import DownstreamModel

os.environ["NCCL_DEBUG"] = "WARN"
LABEL_NUM = {'mosi': 1, 'meld': 7, 'mosei': 1, 'mintrec': 20, 'iemocap': 6}
KEY_METRIC_INDEX = {'mosi': 5, 'meld': 1, 'mosei': 5, 'mintrec': 1, 'iemocap': 1}
model_has_audio_cls = [r"v1\.3\.[3-9]", r"v2\.3", r"v3"]
v2_ckpt = ["v2", "v3.2", "v3.3"]
SYSTEM = "/mnt/ewwe/yts"
SAMPLE_RATE = 16000
CONFIG = "config.json"

if __name__ == '__main__':
    # 1. arguments and config
    parser = argparse.ArgumentParser()
    parser.add_argument('--grad_acc', type=int, default=1)
    parser.add_argument('--audio_path', type=str, default="models/wavlm-base-plus")
    parser.add_argument("--audio_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--dont_show", action="store_true")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--last_conv_layer", type=str, default="no", choices=["no", "layer", "group"])
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument("--multi_audio", action="store_true")
    parser.add_argument("--num_shots", type=int, default=64)
    parser.add_argument('--output_file', type=str, default="results.csv")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pool_size", type=int, default=256)
    parser.add_argument("--pretrain_data", type=int, default=960)
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--system', type=str, default="")
    parser.add_argument("--task", type=str, choices=['iemocap', 'mosi', 'meld', 'mintrec', 'mosei'])
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--text_path', type=str, default="models/roberta-base")
    parser.add_argument('--train_mode', type=str, default="")
    parser.add_argument("--use_turn_ids", action="store_true")
    parser.add_argument('--warmup', type=float, default=0.05)
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
              f" lr {args.lr:.1e} gradacc {args.grad_acc} task {args.task} last_conv_layer {args.last_conv_layer}"
              f" prompt {args.prompt} train_mode {args.train_mode}")
    model_name = model_name.split('-')
    args.audio_path, args.model, args.output_file, args.text_path = map(
        lambda x: os.path.join(SYSTEM, x),
        [args.audio_path, args.model, args.output_file, args.text_path])
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
    config.temperature = args.temperature
    config.pool_size = args.pool_size
    config.num_shots = args.num_shots
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
           f"prompt {args.prompt} bert {turn_embeddings is not None}")
    model = DownstreamModel(args.model, config, label_num, turn_embeddings=turn_embeddings).to(args.device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.lr, bias_correction=False)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale="dynamic")
    # 4.dataset
    c = DataCollatorForDownstream(audio_length, args.task in ["mosi", "mosei"], args.prompt)
    train_data = DownstreamDataset(SYSTEM, args.task, "train", args.train_mode, tokenizer, v2, args.prompt, args.multi_audio)
    valid_loader = DataLoader(dataset=DownstreamDataset(SYSTEM, args.task, "valid", args.train_mode, tokenizer, v2, args.prompt, args.multi_audio), batch_size=args.eval_batch_size, collate_fn=c)
    test_loader = DataLoader(dataset=DownstreamDataset(SYSTEM, args.task, "test", args.train_mode, tokenizer, v2, args.prompt, args.multi_audio), batch_size=args.eval_batch_size, collate_fn=c)
    # 5.preparation
    experiment_data = [[], [], [], []] if args.task in ["mosi", "mosei"] else [[], [], []]
    early_stop_metric = [-10.0, 0.0, 0.0, 0.0] if args.task in ["mosi", "mosei"] else [-10.0, 0.0, 0.0]
    equal = [False for _ in early_stop_metric]
    best_epoch = 0
    best_metrics = []
    for epoch in range(args.epochs):
        # TODO 1 过一遍整个数据集，算出representation，将representation按类别归类，求中心点
        model.eval()
        pre_loader = DataLoader(dataset=train_data, batch_size=args.eval_batch_size, collate_fn=c)
        train_it = pre_loader if args.dont_show else tqdm.tqdm(pre_loader, desc="calculate center")
        repr_by_label = [[] for _ in range(LABEL_NUM[args.task])]  # 从label到其对应所有sample_id及representation的映射
        representations = []  # 从sample_id到representation的映射
        center_by_sample = []  # 从sample_id到center的映射
        labels = []
        sample_id = 0
        with torch.no_grad():
            for (count, batch) in enumerate(train_it):
                batch = {k: (v.to(args.device) if v is not None else None) for k, v in batch.items()}
                representation = model.generate_representations(batch["audio"], batch["text"], batch["aam"],
                                                                batch["tam"], batch["turn_id"], batch["prompt"])
                representations.append(representation)
                representation = [x.squeeze(0) for x in torch.split(representation, 1, dim=0)]
                label = batch["label"].tolist()
                labels.extend(label)
                for i, x in enumerate(label):
                    repr_by_label[x].append(representation[i])
                sample_id += len(label)
            center_by_label = torch.stack([torch.mean(torch.stack(repr_by_label[i], dim=0), dim=0) for i in range(LABEL_NUM[args.task])], dim=0)
            representations = torch.cat(representations, dim=0)
            temp = model
            while hasattr(temp, "module"):
                temp = temp.module
            temp.spcl.set_pool(repr_by_label)
            # TODO 2 将每个样本根据与中心点的相似度排序，选择这个epoch要用到的数据集
            center_by_label_norm = normalize(center_by_label, dim=-1)
            sim = torch.sum(normalize(representations, dim=-1) * center_by_label[torch.LongTensor(labels)], dim=-1)
            sorted_ids = torch.argsort(sim, descending=True).tolist()
            num_candidates_by_labels = [round(len(p) * (epoch + 1) / args.epochs) for p in sample_id_by_labels]
            candidate_sample_ids = [[] for _ in sample_id_by_labels]
            for i in sorted_ids:
                if len(candidate_sample_ids[labels[i]]) < num_candidates_by_labels[labels[i]]:
                    candidate_sample_ids.append(i)
            for i in candidate_sample_ids:
                for j in i:
                    print(j, end=" ")
                print()
        # TODO 3 整理训练集和scheduler
        train_data_for_epoch = EpochDataset([train_data[i] for j in candidate_sample_ids for i in j])
        if n_gpu > 1:
            model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=[args.local_rank])
            train_loader = DataLoader(dataset=train_data_for_epoch, batch_size=args.batch_size, collate_fn=c,
                                      sampler=DistributedSampler(train_data_for_epoch), pin_memory=True)
            train_loader.sampler.set_epoch(epoch)
        else:
            train_loader = DataLoader(dataset=train_data_for_epoch, batch_size=args.batch_size, collate_fn=c, sampler=RandomSampler(train_data_for_epoch))
        steps = (len(train_data_for_epoch) - 1) // args.batch_size // args.grad_acc // max(1, n_gpu) + 1
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup * steps), num_training_steps=steps)
        # TODO 4 常规的训练过程，损失增加SPCL
        model.train()
        epoch_train_loss = []
        time.sleep(1)
        train_it = train_loader if args.dont_show else tqdm.tqdm(train_loader, desc="train")
        for (count, batch) in enumerate(train_it):
            batch = {k: (v.to(args.device) if v is not None else None) for k, v in batch.items()}
            ce_loss, spcl_loss = model(batch["audio"], batch["text"], batch["aam"], batch["tam"], batch["turn_id"],
                                       batch["label"], batch["prompt"])
            loss = ce_loss + spcl_loss
            if n_gpu <= 1:
                epoch_train_loss.append(float(loss.detach().cpu()))
                if not args.dont_show:
                    train_it.set_postfix_str(f"CE: {ce_loss:.4f} SPCL: {spcl_loss:.4f}")
            loss = loss / args.grad_acc
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if (count + 1) % args.grad_acc == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        if not args.dont_show and n_gpu <= 1:
            print(f"Epoch {epoch:03d} average loss {torch.mean(torch.tensor(epoch_train_loss)):.4f}")
        # TODO 5 验证和测试
        model.eval()
        epoch_val_loss = []
        pred_y, true_y = [], []
        with torch.no_grad():
            time.sleep(1)
            valid_it = valid_loader if args.dont_show else tqdm.tqdm(valid_loader, desc="validate")
            for batch in valid_it:
                batch = {k: (v.to(args.device) if v is not None else None) for k, v in batch.items()}
                logits, loss = model(batch["audio"], batch["text"], batch["aam"], batch["tam"], batch["turn_id"],
                                     batch["label"], batch["prompt"], mode="valid")
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
            test_it = test_loader if args.dont_show else tqdm.tqdm(test_loader, desc="test")
            for batch in test_it:
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
        if get_rank() == 0:
            print(early_stop_metric)
        if epoch - best_epoch == args.patience:
            if get_rank() == 0:
                print(f"early stopping at {epoch + 1}")
            break
    if get_rank() == 0:
        params = ['-'.join([model_name[0], model_name[-1]]), str(args.pretrain_data), str(args.batch_size * n_gpu),
                  str(args.epochs), str(args.lr), str(args.grad_acc), str(args.seed), args.task, str(args.warmup),
                  args.last_conv_layer, str(args.multi_audio), str(args.prompt), "no", "3", args.train_mode]
        experiment_data.append(best_metrics)
        with open(args.output_file, "a+") as f:
            f.write("\n" + ",".join(params + [",".join([f"{r}" if isinstance(r, int) else f"{r:.4g}" for r in exp])
                                              for exp in experiment_data]))
