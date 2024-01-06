import json
import deepspeed
from apex import amp
from torch.optim import AdamW
from apex.optimizers import FusedAdam
from torch.nn.functional import normalize
from transformers import get_linear_schedule_with_warmup

from util import *
from parallel import DDP
from tpp.tpp_trainer import ATForTPP
from rm.rm_trainer import ATRewardModel
SAMPLE_RATE = 1600


def step(loss, model, args, mode, optimizer=None, scheduler=None):
    if mode == "ds":
        model.backward(loss)
        model.step()
    else:
        if "amp" in mode:
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


def construct_mean_map(d):
    a = torch.arange(d)
    a = a[None, :] + 1 - a[:, None]
    return torch.clamp_min(a, 1).unsqueeze(-1)


def split_audio_features(audio, audio_attention_mask):
    bs = audio_attention_mask.shape[0] // 2
    audio_attention_mask_sum = torch.clamp_max(audio_attention_mask.view(bs, 2, -1).sum(-1).div(SAMPLE_RATE, rounding_mode='trunc'), 99)
    audio_attention_mask_sum[:, 0] += 1
    audio_attention_mask_sum[:, 1] += (audio_attention_mask_sum[:, 0] + 1)
    audio_attention_mask_sum = audio_attention_mask_sum.long().tolist()
    # print(f"{torch.distributed.get_rank()}, {audio_attention_mask.shape}, {audio_attention_mask.sum(-1)}, {audio.shape}, {audio_attention_mask_sum}")
    audio_features = []
    for i in range(bs):
        audio_features.append(audio[i, 1:audio_attention_mask_sum[i][0]])
        audio_features.append(audio[i, audio_attention_mask_sum[i][0]+1:audio_attention_mask_sum[i][1]])
    return audio_features


def split_text_words(text, split_mark):
    text_words = []
    for t, m in zip(text, split_mark):
        text_words.append(t[:m])
        text_words.append(t[m:])
    return text_words


def precompute_max_sim(s):
    # Precompute the maximum of s(l, i, j) for all i, j and every possible k ≤ l ≤ i
    m, n = s.shape[-2:]
    max_s = torch.zeros_like(s).tolist()
    argmax_s = torch.zeros_like(s).tolist()
    s = s.tolist()
    for j in range(n):
        for i in range(m):
            max_s[0][i][j] = s[0][i][j]
            argmax_s[0][i][j] = 0
            for k in range(1, i + 1):
                if max_s[k-1][i][j] < s[k][i][j]:
                    max_s[k][i][j] = s[k][i][j]
                    argmax_s[k][i][j] = k
                else:
                    max_s[k][i][j] = max_s[k-1][i][j]
                    argmax_s[k][i][j] = k - 1
    return s, max_s, argmax_s


class PPOTrainer:

    def __init__(self, args):
        # super(PPOTrainer, self).__init__()
        self.actor, self.actor_config, self.actor_mode, self.actor_optim, self.actor_scheduler = self.init_trainable(args, ATForTPP, args.actor_path, args.actor_lr)
        self.ref, _ = self.init_ref(args, ATForTPP, args.actor_path)
        # self.critic, self.critic_optim, self.critic_scheduler = self.init_trainable(args, ATRewardModel, args.reward_path)
        self.reward, self.reward_config = self.init_ref(args, ATRewardModel, args.reward_path)
        
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

    def compute_alignment(self, audio, text, audio_mask, text_mask, turn_id, text_valid, split_marks):
        audio_features, text_words = self.reward.forward_features_for_ppo(audio, text, audio_mask, text_mask, turn_id, text_valid=text_valid)
        audio_features = split_audio_features(audio_features, audio_mask)
        text_words = split_text_words(text_words, split_marks)  # audio和text均为2B个
        tpp_starts, tpp_ends = [], []  # 一个batch内部所有label打成一个1D数组
        for a, t in zip(audio_features, text_words):
            # a: M*H t: N*H
            m = a.shape[0]
            n = t.shape[0]
            a = a.unsqueeze(1).repeat(1, m, 1).permute(2, 0, 1).triu().permute(1, 2, 0) / self.mean_map.to(audio.device)[:m, :m, :]
            sim = torch.einsum("ijk,lk->ijl", normalize(a), normalize(t))  # M*M*N
            sim, max_sim, argmax_sim = precompute_max_sim(sim)
            # 动态规划
            obj = [[float("-inf") for _ in range(n)] for _ in range(m)]
            starts = [[0 for _ in range(n)] for _ in range(m)]  # 每次被选中的argmax_sim，由end_labels[i]指向start_labels[i]
            ends = [[0 for _ in range(n)] for _ in range(m)]  # 由end_labels[i]指向end_labels[i-1]
            for j in range(n):  # 前j个词
                for i in range(m):  # 前i个语音token
                    if j == 0:
                        obj[i][j] = sim[0][i][j]
                        starts[i][j] = 0
                        ends[i][j] = -1
                    else:
                        for k in range(i):
                            if obj[k][j-1] + max_sim[k][i][j] > obj[i][j]:
                                obj[i][j] = obj[k][j-1] + max_sim[k][i][j]
                                starts[i][j] = argmax_sim[k][i][j]
                                ends[i][j] = k
            # 最大匹配值
            mm, argmax = float("-inf"), -1
            for i, o in enumerate(obj):
                if mm < o[-1]:
                    mm = o[-1]
                    argmax = i
            # 回溯路径
            s, e = [], [argmax]
            for j in range(n - 1, -1, -1):
                if j > 0:
                    e.append(ends[e[-1]][j])
                s.append(starts[e[-1]][j])
            tpp_starts.extend(s[::-1])
            tpp_ends.extend(e[::-1])

        if self.args.num_ends == 1:
            tpp_starts = (torch.tensor(tpp_starts, device=audio.device) / 100).to(dtype=audio.dtype)
            tpp_ends = (torch.tensor(tpp_ends, device=audio.device) / 100).to(dtype=audio.dtype)
            return tpp_starts, tpp_ends
        return torch.LongTensor(tpp_starts), torch.LongTensor(tpp_ends)

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
