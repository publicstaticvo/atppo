import os
import sys
import tqdm
import json
import math
import argparse
import deepspeed
import numpy as np
import datetime

print(datetime.datetime.now())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from util import *
from rm_trainer import ATRewardModel
from rm_dataloader import RMDataset, DataCollatorForRM
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import RobertaTokenizerFast, AdamW, get_linear_schedule_with_warmup

sw = None
SAMPLE_RATE = 16000
CONFIG = "config.json"
os.environ["NCCL_DEBUG"] = "WARN"

if __name__ == "__main__":
    # 1.输入参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--apex_level", default=0, type=int)
    parser.add_argument("--actor_lr", default=1e-5, type=float)
    parser.add_argument("--actor_path", default=None, type=str, required=True)
    parser.add_argument("--audio_length", default=10, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--critic_lr", default=1e-5, type=float)
    parser.add_argument("--critic_path", default=None, type=str, required=True)
    parser.add_argument("--dont_show", action='store_true')
    parser.add_argument("--ds_config", default=None, type=str)
    parser.add_argument("--ds_stage", default=3, type=int)
    parser.add_argument("--file_prefix", default=None, type=str)
    parser.add_argument("--grad_acc", default=16, type=int)
    parser.add_argument("--grad_norm", default=0., type=float)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--loss_scale", default=0., type=float)
    parser.add_argument("--model_name", default="v1.1", type=str)
    parser.add_argument("--model_save_path", default=None, type=str)
    parser.add_argument("--num_turns", default=8, type=int)
    parser.add_argument("--num_fused_layers", default=1, type=int)
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--save_tmp", default=None, type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--text_length", default=512, type=int)
    parser.add_argument("--tokenizer_path", default=None, type=str, required=True)
    parser.add_argument("--train_epochs", default=10, type=int)
    parser.add_argument("--transcripts", default=None, type=str, required=True)
    parser.add_argument("--warmup", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
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
    # 3。使用tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_path)
    # 4。读输入数据
    train_data = RMDataset(args.transcripts, args.num_turns, args.file_prefix)
    # 5。整理config并建立模型
    # 6。数据并行
    c = DataCollatorForRM(tokenizer, config, args.apex_level > 0)
    if args.local_rank >= 0:
        num_train_steps = math.ceil(num_train_steps / n_gpu)
        train_loader = DataLoader(train_data, sampler=DistributedSampler(train_data, seed=args.seed), batch_size=args.batch_size, collate_fn=c, pin_memory=True, num_workers=20)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=c, sampler=RandomSampler(train_data))
    model.train()
    losses = []
    outer_it = tqdm.trange(args.train_epochs)
    for i in outer_it:
        inner_it = train_loader if args.dont_show or get_rank() else tqdm.tqdm(train_loader, desc="Inner")
        le = len(inner_it)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(i)
        losses = [0, 0]
        for j, batch in enumerate(inner_it):
            a_input, a_mask, a_valid, t_input, t_mask, t_valid, turn_id, neg, mlm_labels = batch
            a_input, a_mask, t_input, t_mask, turn_id = map(lambda x: x.to(args.device), [a_input, a_mask, t_input, t_mask, turn_id])
            a_valid, t_valid, neg = map(lambda x: [t.to(args.device) for t in x], [a_valid, t_valid, neg])
            if mlm_labels is not None: mlm_labels.to(args.device)
            mlm, mam, rm = model(a_input, t_input, a_mask, t_mask, turn_id, a_valid, t_valid, neg, mlm_labels)
            loss = mlm + mam + rm
            losses[0] += float(loss)
            losses[1] += float(rm)
            if not args.dont_show and get_rank() == 0:
                inner_it.set_postfix_str(f"loss: {loss:.4f}|rm: {rm:.4f}")
            loss = loss / args.grad_acc
            if args.ds_config:
                model.backward(loss)
                model.step()
            else:
                if args.apex_level > 0:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if args.grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.grad_norm)
                else:
                    loss.backward()
                    if args.grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                if (j + 1) % args.grad_acc == 0 or j + 1 == le:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
        if get_rank() == 0 and ((i + 1) % args.save_interval == 0 or args.save_tmp) and args.model_save_path:
            save_path = os.path.join(args.model_save_path, f"{args.model_name}-{i + 1}" if (i + 1) % args.save_interval == 0 else args.save_tmp)
            temp = model
            while hasattr(temp, "module"):
                temp = temp.module
            temp.save_pretrained(save_path)
        if get_rank() == 0:
            outer_it.set_postfix_str(f"loss: {losses[0] / le:.4f}|rm:{losses[1] / le:.4f}")