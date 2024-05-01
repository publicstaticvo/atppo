import os
import sys
import math
import tqdm
import argparse
import numpy as np
import datetime

print(datetime.datetime.now())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from util import *
from SPECTRA import TPPDataset
from dp_trainer import DPTrainer
from dp_dataset import DPDataset, DataCollatorForDP
from transformers import RobertaTokenizerFast
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

sw = None
SAMPLE_RATE = 16000
CONFIG = "config.json"
os.environ["NCCL_DEBUG"] = "WARN"

if __name__ == "__main__":
    # 1.输入参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--apex_level", default=0, type=int)
    parser.add_argument("--audio_length", default=10, type=float)
    parser.add_argument("--audio_path", default=None, type=str, required=True)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--dont_show", action='store_true')
    parser.add_argument("--ds_config", default=None, type=str)
    parser.add_argument("--ds_stage", default=3, type=int)
    parser.add_argument("--file_prefix", default=None, type=str)
    parser.add_argument("--grad_acc", default=0, type=int)
    parser.add_argument("--grad_norm", default=0., type=float)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--loss_scale", default=0., type=float)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--model_name", default="v1.1", type=str)
    parser.add_argument("--model_save_path", default=None, type=str)
    parser.add_argument("--num_turns", default=8, type=int)
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--save_tmp", default=None, type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--text_length", default=512, type=int)
    parser.add_argument("--text_path", default=None, type=str, required=True)
    parser.add_argument("--train_epochs", default=10, type=int)
    parser.add_argument("--transcripts", default=None, type=str, required=True)
    parser.add_argument("--warmup", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if args.grad_acc == 0:
        args.grad_acc = 1
    else:
        args.grad_acc *= 3
    if not torch.cuda.is_available():
        args.apex_level = 0
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", init_method='env://')
    # 2.设随机数
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # 3。读输入数据
    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_path)
    train_data = DPDataset(args.transcripts, args.num_turns, args.file_prefix)
    args.num_train_steps = args.train_epochs * math.ceil(len(train_data) / args.batch_size / args.grad_acc)
    # 4。建立模型
    args.batch_size *= 2
    trainer = DPTrainer(args, tokenizer)
    c = DataCollatorForDP(args, tokenizer)
    if args.local_rank >= 0:
        train_loader = DataLoader(train_data, sampler=DistributedSampler(train_data, seed=args.seed), batch_size=args.batch_size, collate_fn=c, pin_memory=True, num_workers=20)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=c, sampler=RandomSampler(train_data))
    trainer.train()
    losses = []
    outer_it = tqdm.trange(args.train_epochs)
    for i in outer_it:
        inner_it = train_loader if args.dont_show or get_rank() else tqdm.tqdm(train_loader, desc="Inner")
        le = len(inner_it)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(i)
        losses = [0, 0, 0, 0]
        for j, batch in enumerate(inner_it):
            batch = {k: v if k == "splits" else v.to(args.device) for k, v in batch.items()}
            *_, rs, step1, step3, step5 = trainer.train_dp(**batch)
            losses[0] += float(step1)
            losses[1] += float(step3)
            losses[2] += float(rs)
            losses[3] += float(step5)
            if not args.dont_show and get_rank() == 0:
                inner_it.set_postfix_str(f"step1: {step1:.4f}|step3: {step3:.4f}|step5: {step5:.4f}|rs: {rs:.4f}")
        if get_rank() == 0 and ((i + 1) % args.save_interval == 0 or args.save_tmp) and args.model_save_path:
            trainer.save_pretrained(os.path.join(args.model_save_path, f"{args.model_name}-{i + 1}" if (i + 1) % args.save_interval == 0 else args.save_tmp))
        if get_rank() == 0:
            outer_it.set_postfix_str(f"step1: {losses[0] / le:.4f}|step3:{losses[1] / le:.4f}|step5: {losses[3] / le:.4f}|rs: {losses[2] / le:.4f}")
