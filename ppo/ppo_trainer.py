import json
import deepspeed
from apex import amp
from torch.optim import AdamW
from apex.optimizers import FusedAdam
from torch.nn.functional import normalize
from transformers import get_linear_schedule_with_warmup

from util import *
from tpp.tpp_trainer import ATForTPP
from models import ATRewardWord
SAMPLE_RATE = 1600


def step(loss, model, args, optimizer=None, scheduler=None):
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
        if model.step_count % args.grad_acc == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


class PPOTrainer:

    def __init__(self, args):
        # super(PPOTrainer, self).__init__()
        self.actor, self.actor_config, self.actor_mode, self.actor_optim, self.actor_scheduler = self.init_trainable(args, ATForTPP, args.actor_path, args.actor_lr)
        self.ref, _ = self.init_ref(args, ATForTPP, args.actor_path)
        # self.critic, self.critic_optim, self.critic_scheduler = self.init_trainable(args, ATRewardModel, args.reward_path)
        self.reward, self.reward_config = self.init_ref(args, ATRewardWord, args.reward_path)
        
        self.args = args
        self.perform_mlm = True
        self.num_ends = self.actor_config.fused.num_ends
        self.hidden_size = self.actor_config.text.hidden_size
        self.kl_ctl = 0.1
        self.clip_reward_value = 10
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.experience_count = 0
        self.mean_map = construct_mean_map(200).half()

    def kl(self, actor, ref, eps=1e-4):
        if self.args.num_ends == 1:
            return torch.mean(torch.pow(actor - ref, 2))
        return torch.mean(actor * torch.log(actor / torch.clamp_min(ref, eps)))

    def valid_filter(self, outputs, valid, pooling_mode):
        words = valid.shape[0]
        if pooling_mode == "first":
            # 每个valid形状为L
            return outputs.masked_select(valid.unsqueeze(-1)).view(-1, self.hidden_size)
        elif pooling_mode == "mean":
            # 每个valid形状为Ni*L
            temp = outputs.unsqueeze(0).repeat(words, 1, 1).masked_fill(~valid.unsqueeze(-1), 0)
            return torch.sum(temp, dim=1) / torch.sum(valid, dim=1, keepdim=True)

    def train_ppo(self, audio_input, audio_mask, full_text, text_input, text_mask, mlm_label=None, turn_id=None, start_valid=None, end_valid=None, splits=None):
        # 1 标注
        with torch.no_grad():
            bs, text_len = mlm_label.shape
            positive_audio_input = audio_input.view(bs, 3, -1)[:, :2].contiguous().view(bs * 2, -1)
            positive_audio_mask = audio_mask.view(bs, 3, -1)[:, :2].contiguous().view(bs * 2, -1)
            positive_text_mask = text_mask.view(bs, 2, -1)[:, 0]
            positive_turn_id = turn_id.view(bs, 2, -1)[:, 0]
            tpp_starts, tpp_ends = self.compute_alignment(positive_audio_input, full_text, positive_audio_mask, positive_text_mask, positive_turn_id, start_valid, splits)
        # 2 求偏差，actor step
        fused_features, mam_label, a_masked = self.actor.model(audio_input, text_input, audio_mask, text_mask, turn_id, self.perform_mlm)
        fused_features = fused_features.view(bs, 4, -1, self.hidden_size)
        text_fused = fused_features[:, 0, :text_len]
        start_valid, end_valid = map(lambda x: torch.cat(x).view(bs, -1), [start_valid, end_valid])
        mlm = self.actor.mlm_loss(text_fused, mlm_label)
        mam = self.actor.mam_loss(fused_features[:, 0, text_len:], mam_label, a_masked)
        rs_loss = self.actor.response_selection(fused_features, bs)
        span_loss, pred_actor = self.actor.tpp_loss(text_fused, start_valid, end_valid, tpp_starts, tpp_ends)
        with torch.no_grad():
            pred_ref = self.ref.tpp_loss(text_fused, start_valid, end_valid)
        kl = self.kl(pred_actor, pred_ref)
        loss = mlm + mam + rs_loss + span_loss * 3 + self.kl_ctl * kl
        step(loss, self.actor, self.args, self.actor_mode, self.actor_optim, self.actor_scheduler)
        return mlm, mam, rs_loss, span_loss, kl, loss

    def init_trainable(self, args, model_class, model_name, lr):
        # 1 ds config
        if args.ds_config == "default":
            args.ds_config = get_train_ds_config(args.batch_size, torch.cuda.device_count(), args.grad_acc, args.ds_stage, args.apex_level)
        elif args.ds_config:
            with open(args.ds_config, "w+") as f:
                args.ds_config = json.load(f)
        # 2 model
        model = model_class.from_pretrained(model_name)
        config = model.config
        # 3 optimizer
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        no_decay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        ogp = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        if args.apex_level > 0:
            optimizer = FusedAdam(ogp, lr=lr, bias_correction=False)
        else:
            optimizer = AdamW(ogp, lr=lr, eps=1e-8)
        warmup_steps = int(args.warmup * args.num_train_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.num_train_steps)
        # 4 parallel
        if args.ds_config:
            model, optimizer, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer, config=args.ds_config, lr_scheduler=scheduler, dist_init_required=True)
            model_mode = "ds"
        else:
            model.to(args.device)
            model_mode = ""
            if args.apex_level > 0:
                model, optimizer = amp.initialize(model, optimizer, opt_level=f"O{args.apex_level}", keep_batchnorm_fp32=False if args.apex_level >= 2 else None, loss_scale="dynamic" if args.loss_scale == 0. else args.loss_scale)
                model_mode = "amp"
            if args.local_rank >= 0:
                model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=[args.local_rank])
                model_mode += "ddp"
        return model, config, model_mode, optimizer, scheduler

    def init_ref(self, args, model_class, model_name):
        # 1 ds config
        if args.ds_config:
            args.ds_config = get_eval_ds_config(args.batch_size, args.ds_stage, args.apex_level)
        # 2 model
        model = model_class.from_pretrained(model_name)
        config = model.config
        # 3 parallel
        if args.ds_config:
            model, *_ = deepspeed.initialize(model=model, config=args.ds_config, dist_init_required=True)
        else:
            model.to(args.device)
            if args.apex_level > 0:
                model = amp.initialize(model, opt_level=f"O{args.apex_level}", keep_batchnorm_fp32=False if args.apex_level >= 2 else None)
            if args.local_rank >= 0:
                model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=[args.local_rank])
        return model, config

    def save_pretrained(self, path):
        temp = self.actor
        while hasattr(temp, "module"):
            temp = temp.module
        temp.save_pretrained(path)

    def train(self):
        self.actor.train()

    def eval(self):
        self.actor.eval()
        self.ref.eval()
        self.reward.eval()
