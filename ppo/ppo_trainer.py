import json
import deepspeed
import torch
from apex import amp
from torch.optim import AdamW
from apex.optimizers import FusedAdam
from torch.nn.functional import normalize
from transformers import get_linear_schedule_with_warmup
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from util import *
from torch import nn
from actor_model import ActorModel
from models import ATSingleTurnModel, ATSingleTurnForSentenceAlign, WavLMMAMHead
SAMPLE_RATE = 1600


class PPOTrainer:

    def __init__(self, args):
        self.actor, self.actor_config, self.actor_optim, self.actor_scheduler = self.init_trainable(args, ActorModel, args.actor_path, args.actor_lr)
        self.ref, _ = self.init_ref(args, ActorModel, args.actor_path)
        self.critic, self.critic_config, self.critic_optim, self.critic_scheduler = self.init_trainable(args, ATSingleTurnModel, args.critic_path, args.critic_lr)
        self.reward, self.reward_config = self.init_ref(args, ATSingleTurnForSentenceAlign, args.reward_path)

        self.hidden_size = self.actor_config.text.hidden_size
        self.mlm_head = RobertaLMHead(self.actor_config.text)
        self.mam_head = WavLMMAMHead(self.hidden_size, self.actor_config.audio.conv_dim[-1])
        self.selection_head = nn.Linear(self.hidden_size, 4)
        self.ce = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()
        
        self.args = args
        self.num_ends = self.actor_config.fused.num_ends
        self.vocab_size = self.actor_config.text.vocab_size
        self.max_audio_length = self.actor_config.audio.max_length
        self.temperature = 1
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95

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

    def train_ppo(self, audio_input, audio_mask, text_input, text_mask, mlm_label=None, turn_id=None,
                  start_valid=None, end_valid=None, splits=None, full_text_for_rm=None,
                  full_text_for_rm_mask=None, start_valid_for_rm=None):
        """
            audio_input = 3B * 160000; audio_mask = 3B * 160000
            text_input = 2B * 8轮ML; text_mask = 2B * 8轮ML; mlm_label = B * 8轮ML; turn_id=2*ML
            full_text_for_rm = 2B * 一轮ML; full_text_for_rm_mask = 2B * 一轮ML
            start_valid = B * 8轮ML; end_valid = B * 8轮ML
            turn_id = 2B * 8轮ML
            start_valid_for_rm = 2B大小列表
            splits: B*2大小的列表，代表第7轮、第8轮文本中词的个数。用splits配合actor_start可分离第7轮和第8轮的词语。
            综上所述：offsets将text_feature切分为history, anchor和positive
                    start_valid和end_valid从text_feature获得anchor和positive的词位置
                    start_valid_for_rm和end_valid_for_rm从词级别RM获取词位置
                    splits将actor_start和actor_end分成两轮
        """
        # 1 进行预测，获得actor预测结果和ref预测结果
        bs, text_len = mlm_label.shape
        device = mlm_label.device
        with torch.no_grad():
            actor_start, actor_end, *_ = self.actor(audio_input, text_input, audio_mask, text_mask, turn_id=turn_id, start_valid=start_valid, end_valid=end_valid)
            start_valid, end_valid = map(torch.cat, [start_valid, end_valid])
            ref_start, ref_end, *_ = self.ref(audio_input, text_input, audio_mask, text_mask, turn_id=turn_id, start_valid=start_valid, end_valid=end_valid)
            # 2 求得RM
            audio_input_for_rm = audio_input.view(bs, 3, -1)[:, :2].contiguous().view(bs * 2, -1)
            audio_mask_for_rm = audio_mask.view(bs, 3, -1)[:, :2].contiguous()
            audio_length = audio_mask.sum(-1).tolist()
            # 2.1 根据分类结果重新分割音频
            new_audio, audio_valid = [], []
            for i, (s, e, ml, o) in enumerate(zip(actor_start, actor_end, audio_length, splits)):  # B次循环
                s = (s.max(-1).values.long() * 1600)  # N * 100 => N
                e = (e.max(-1).values.long() * 1600)
                s1, s2 = s[:o[0]], s[o[0]:]
                e1, e2 = e[:o[0]], e[o[0]:]
                s1 = s1.clamp_max(ml[0] - 1600)
                s2 = s2.clamp_max(ml[1] - 1600)
                e1 = e1.clamp(s1 + 1600, ml[0])
                e2 = e2.clamp(s2 + 1600, ml[1])
                new_audio.append(concat_audio(audio_input_for_rm[2 * i], torch.stack([s1, e1], dim=-1)))
                new_audio.append(concat_audio(audio_input_for_rm[2 * i + 1], torch.stack([s2, e2], dim=-1)))
                s1_labels, s2_labels, e1_labels, e2_labels = map(lambda x: x.div(1600, rounding_mode="floor").tolist(), [s1, s2, e1, e2])
                audio_valid_1 = torch.zeros((len(s1_labels), self.num_ends), device=device, dtype=torch.bool)
                for j, (k, l) in enumerate(zip(s1_labels, e1_labels)):  # N次循环
                    audio_valid_1[j, k:l] = True
                audio_valid.append(audio_valid_1)
                audio_valid_2 = torch.zeros((len(s2_labels), self.num_ends), device=device, dtype=torch.bool)
                for j, (k, l) in enumerate(zip(s2_labels, e2_labels)):
                    audio_valid_1[j, k:l] = True
                audio_valid.append(audio_valid_2)
            # 2.2 pad
            ml = self.max_audio_length * 2
            new_audio_mask = []
            for i in range(len(new_audio)):  # 2B次循环
                if new_audio[i].shape[0] <= ml:
                    padding = torch.zeros(ml - new_audio[i].shape[0], dtype=new_audio[i].dtype, device=new_audio[i].device)
                    new_audio_mask.append(torch.ones_like(new_audio[i], dtype=torch.long))
                    new_audio[i] = torch.cat([new_audio[i], padding])
                else:
                    new_audio[i] = new_audio[i][:ml]
                    new_audio_mask.append(torch.ones(ml, dtype=torch.long))
            new_audio, new_audio_mask = map(lambda x: x.stack(0), [new_audio, new_audio_mask])
            # 2.3 reward
            reward_scores = self.reward(new_audio, full_text_for_rm, new_audio_mask, full_text_for_rm_mask).clamp(-self.clip_reward_value, self.clip_reward_value).tolist()
            # 2.4 value
            audio_features, _, text_features = self.critic(audio_input_for_rm, full_text_for_rm, audio_mask_for_rm.view(bs * 2, -1), full_text_for_rm_mask)
            values = []
            for a, t, av, tv in zip(audio_features, text_features, audio_valid, start_valid_for_rm):  # 2B次循环
                audio_words = self.valid_filter(a, av, self.critic_config.audio.pooling_mode)
                text_words = self.valid_filter(t, tv, self.critic_config.text.pooling_mode)
                sim = similarity(text_words, audio_words, self.temperature, mode="diag_cosine")  # N
                values.append(sim)
            values = torch.cat(sim)
            assert values.shape == start_valid.shape

        # 3 actor step
        # 假如actor_start_max为1D的tensor?
        actor_start, actor_end, fused_features, text_fused, mam_label, a_masked = self.actor(audio_input, text_input, audio_mask, text_mask, mlm_label=mlm_label, turn_id=turn_id, start_valid=start_valid, end_valid=end_valid)
        fused_features = fused_features.view(bs, 4, -1, self.hidden_size)
        # compute_rewards
        actor_start_max, actor_end_max, ref_start_max, ref_end_max = map(lambda x: x.max(-1).values, [actor_start, actor_end, ref_start, ref_end])
        actor_log_probs = (actor_start_max * actor_end_max).log()
        ref_log_probs = (ref_start_max * ref_end_max).log()
        rewards = -self.kl_ctl * (actor_log_probs - ref_log_probs)  # KL
        splits = [x for s in splits for x in s]
        for r, o in zip(reward_scores, splits):
            rewards[o - 1] += r
        # get_advantages_and_returns
        with torch.no_grad:
            lastgaelam = 0
            advantages = [0 for _ in range(values.shape[0])]
            for i, o in enumerate(splits):
                for t in reversed(range(0 if i == 0 else splits[i - 1], o)):
                    next_values = values[t + 1] if t < o - 1 else 0.0
                    delta = rewards[t] + self.gamma * next_values - values[t]
                    lastgaelam = delta + self.gamma * self.lam * lastgaelam
                    advantages[t] = lastgaelam
            advantages = torch.tensor(advantages, dtype=fused_features.dtype, device=device)
            returns = advantages + values
        # actor_loss
        ratio = actor_start_max * actor_end_max / ref_start_max / ref_end_max
        ppo_loss1 = -advantages * ratio
        ppo_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        ppo_loss = torch.max(ppo_loss1, ppo_loss2).mean()
        # MLM, MAM, CRS
        mlm = self.actor.mlm_loss(text_fused, mlm_label)
        mam = self.actor.mam_loss(fused_features[:, 0, text_len:], mam_label, a_masked)
        rs_loss = self.actor.response_selection(fused_features, bs)
        actor_loss = mlm + mam + rs_loss + ppo_loss
        step(actor_loss, self.actor, self.args, self.actor_optim, self.actor_scheduler)
        # 4 critic step
        audio_features, _, text_features = self.critic(audio_input_for_rm, full_text_for_rm, audio_mask_for_rm.view(bs * 2, -1), full_text_for_rm_mask)
        values = []
        for a, t, av, tv in zip(audio_features, text_features, audio_valid, start_valid_for_rm):  # 2B次循环
            audio_words = self.valid_filter(a, av, self.critic_config.audio.pooling_mode)
            text_words = self.valid_filter(t, tv, self.critic_config.text.pooling_mode)
            sim = similarity(text_words, audio_words, self.temperature, mode="diag_cosine")  # N
            values.append(sim)
        values = torch.cat(values)
        critic_loss = ((values - returns) ** 2).mean()
        step(critic_loss, self.critic, self.args, self.critic_optim, self.critic_scheduler)
        return mlm, mam, rs_loss, ppo_loss, actor_loss, critic_loss

    def mlm_loss(self, text, label):
        mlm_pre = self.mlm_head(text)
        mlm_loss = self.ce(mlm_pre.view(-1, self.vocab_size), label.view(-1))  # 未mask的位置，label为-100。
        if torch.isnan(mlm_loss):
            mlm_loss = torch.tensor(0.0, device=text.device)
        return mlm_loss

    def mam_loss(self, audio, label, masked_indices):
        mam_pre = self.mam_head(audio)
        mam_loss = torch.tensor(0.0, device=audio.device)
        if torch.sum(masked_indices[1]) != 0:
            mam_loss = self.l1(mam_pre.masked_select(masked_indices[0].unsqueeze(-1)), label.masked_select(masked_indices[1]))
        return mam_loss

    def response_selection(self, fused_input, batch_size):
        response_select = self.selection_head(fused_input[:, :, 0].view(4 * batch_size, self.hidden_size))
        rs_loss = self.ce(response_select, torch.arange(4).to(fused_input.device).repeat(batch_size))
        return rs_loss

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
        else:
            model.to(args.device)
            if args.apex_level > 0:
                model, optimizer = amp.initialize(model, optimizer, opt_level=f"O{args.apex_level}", keep_batchnorm_fp32=False if args.apex_level >= 2 else None, loss_scale="dynamic" if args.loss_scale == 0. else args.loss_scale)
            if args.local_rank >= 0:
                model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=[args.local_rank])
        return model, config, optimizer, scheduler

    def init_ref(self, args, model_class, model_name):
        # 1 ds config
        if args.ds_config:
            args.ds_config = get_eval_ds_config(args.batch_size, 0, args.apex_level)
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
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.ref.eval()
        self.critic.eval()
        self.reward.eval()
