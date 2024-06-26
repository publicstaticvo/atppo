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
from tpp_trainer import TPPTrainer
from tpp_dataset import TPPDataset, DataCollatorForTPP
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
    parser.add_argument("--audio_length", default=10, type=float)
    parser.add_argument("--audio_path", default=None, type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--dont_show", action='store_true')
    parser.add_argument("--ds_config", default=None, type=str)
    parser.add_argument("--ds_stage", default=2, type=int)
    parser.add_argument("--file_prefix", default=None, type=str)
    parser.add_argument("--grad_acc", default=16, type=int)
    parser.add_argument("--grad_ckpt", action='store_true')
    parser.add_argument("--grad_norm", default=0., type=float)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--loss_scale", default=0., type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--model_name", default="v1.1", type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--model_save_path", default=None, type=str)
    parser.add_argument("--no_pretrain", action='store_true')
    parser.add_argument("--num_ends", default=1, type=int)
    parser.add_argument("--num_turns", default=8, type=int)
    parser.add_argument("--num_fused_layers", default=1, type=int)
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--save_tmp", default=None, type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--text_length", default=512, type=int)
    parser.add_argument("--text_path", default=None, type=str)
    parser.add_argument("--train_epochs", default=10, type=int)
    parser.add_argument("--transcripts", default=None, type=str, required=True)
    parser.add_argument("--warmup", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    args = parser.parse_args()
    n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        args.apex_level = 0
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", init_method='env://')
    if args.ds_config == "default":
        args.ds_config = get_train_ds_config(args.batch_size, n_gpu, args.grad_acc, args.ds_stage, args.apex_level)
    elif args.ds_config:
        with open(args.ds_config, "w+") as f:
            args.ds_config = json.load(f)
    # 2.设随机数
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # 3。使用tokenizer
    if args.model_path:
        config = ATConfig.from_pretrained(args.model_path)
    else:
        config = ATConfig.from_json_files(os.path.join(args.audio_path, CONFIG), os.path.join(args.text_path, CONFIG))
        config.set_length(int(args.audio_length * SAMPLE_RATE), args.text_length)
        config.fused.num_hidden_layers = args.num_fused_layers
        config.fused.num_ends = args.num_ends
    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_path)
    # 4。读输入数据
    train_data = TPPDataset(args.transcripts, args.num_turns, args.file_prefix)
    # 5。整理config并建立模型
    if args.no_pretrain:
        model = TPPTrainer(config)
    elif args.model_path:
        model = TPPTrainer.from_pretrained(args.model_path, config=config)
    else:
        model = TPPTrainer(config, args.audio_path, args.text_path)
    # 6。数据并行
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    ogp = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    num_train_steps = args.train_epochs * math.ceil(len(train_data) / args.batch_size / args.grad_acc)
    if args.apex_level > 0:
        from apex import amp
        from apex.optimizers import FusedAdam
        optimizer = FusedAdam(ogp, lr=args.lr, bias_correction=False)
    else:
        optimizer = AdamW(ogp, lr=args.lr, eps=1e-8)
    warmup_steps = int(args.warmup * num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    c = DataCollatorForTPP(tokenizer, config, args.apex_level > 0)
    if args.ds_config:
        model, optimizer, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer, config=args.ds_config,
                                                              lr_scheduler=scheduler, dist_init_required=True)
    else:
        model.to(args.device)
        if args.apex_level > 0:
            model, optimizer = amp.initialize(model, optimizer, opt_level=f"O{args.apex_level}",
                                              keep_batchnorm_fp32=False if args.apex_level >= 2 else None,
                                              loss_scale="dynamic" if args.loss_scale == 0. else args.loss_scale)
        if args.local_rank >= 0:
            model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=[args.local_rank])
    if args.local_rank >= 0:
        num_train_steps = math.ceil(num_train_steps / n_gpu)
        train_loader = DataLoader(train_data, sampler=DistributedSampler(train_data, seed=args.seed), batch_size=args.batch_size, collate_fn=c, pin_memory=True, num_workers=20)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=c, sampler=RandomSampler(train_data), num_workers=20)
    if args.grad_ckpt:
        if isinstance(model, DDP):
            model.module.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()
    model.train()
    losses = []
    outer_it = tqdm.trange(args.train_epochs)
    for i in outer_it:
        inner_it = train_loader if args.dont_show or get_rank() else tqdm.tqdm(train_loader, desc="Inner")
        le = len(inner_it)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(i)
        losses = []
        for j, batch in enumerate(inner_it):
            batch = to_device(batch, args.device)
            mlm_loss, mam_loss, rs_loss, span_loss = model(**batch)
            loss = mlm_loss + mam_loss + rs_loss + span_loss
            if args.num_ends == 1:
                loss += 2 * span_loss
            if not args.dont_show and get_rank() == 0:
                inner_it.set_postfix_str(f"MLM: {mlm_loss:.4f} MAM: {mam_loss:.4f} R-S: {rs_loss:.4f} SPAN: {span_loss:.4f}")
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
            outer_it.set_postfix_str(f"MLM: {mlm_loss:.4f} MAM: {mam_loss:.4f} R-S: {rs_loss:.4f} SPAN: {span_loss:.4f}")
