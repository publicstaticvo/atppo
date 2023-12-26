import json
import math
import torch
import deepspeed
from torch import nn
from apex import amp
from apex.optimizers import FusedAdam
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP

from util import *
from tpp.tpp_trainer import ATForTPP
from rm.rm_trainer import ATRewardModel


class PPOTrainer:

    def __init__(self, args, actor, critic):
        # super(PPOTrainer, self).__init__()
        self.actor, self.actor_optim, self.actor_scheduler = self.init_trainable(args, ATForTPP, actor)
        self.ref = self.init_ref(args, ATForTPP, actor)
        self.critic, self.critic_optim, self.critic_scheduler = self.init_trainable(args, ATRewardModel, critic)
        self.reward = self.init_ref(args, ATRewardModel, critic)

        self.args = args
        self.kl_ctl = 0.1
        self.clip_reward_value = 10
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.experience_count = 0

    def generate_trajectory(self):
        predict = self.actor()
        old_predict = self.ref()
        return predict, old_predict

    def compute_rewards(self):
        features = self.reward()

    def train_ppo(self):
        pass

    def init_trainable(self, args, model_class, model_name):
        # 1 ds config
        if args.ds_config == "default":
            args.ds_config = get_train_ds_config(args.batch_size, torch.cuda.device_count(), args.grad_acc, args.ds_stage, args.apex_level)
        elif args.ds_config:
            with open(args.ds_config, "w+") as f:
                args.ds_config = json.load(f)
        # 2 model
        # TODO: config.audio.train_mode = 2 / 3 验证
        model = model_class.from_pretrained(model_name)
        # 3 optimizer
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        no_decay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        ogp = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        if args.apex_level > 0:
            optimizer = FusedAdam(ogp, lr=args.lr, bias_correction=False)
        else:
            optimizer = AdamW(ogp, lr=args.lr, eps=1e-8)
        warmup_steps = int(args.warmup * args.num_train_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.num_train_steps)
        # 4 parallel
        if args.ds_config:
            model, optimizer, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer, config=args.ds_config, lr_scheduler=scheduler, dist_init_required=True)
        else:
            model.to(args.device)
            if args.apex_level > 0:
                model, optimizer = amp.initialize(model, optimizer, opt_level=f"O{args.apex_level}", keep_batchnorm_fp32=False if args.apex_level >= 2 else None, loss_scale="dynamic" if args.loss_scale == 0. else args.loss_scale)
            if args.local_rank >= 0:
                model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=[args.local_rank])
        return model, optimizer, scheduler

    def init_ref(self, args, model_class, model_name):
        # 1 ds config
        if args.ds_config:
            args.ds_config = get_eval_ds_config(args.batch_size, args.ds_stage, args.apex_level)
        # 2 model
        model = model_class.from_pretrained(model_name)
        # 3 parallel
        if args.ds_config:
            model, *_ = deepspeed.initialize(model=model, config=args.ds_config, dist_init_required=True)
        else:
            model.to(args.device)
            if args.apex_level > 0:
                model = amp.initialize(model, opt_level=f"O{args.apex_level}", keep_batchnorm_fp32=False if args.apex_level >= 2 else None)
            if args.local_rank >= 0:
                model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=[args.local_rank])
        return model

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.ref.eval()
        self.reward.eval()
