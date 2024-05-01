import json
import deepspeed
import torch
from apex import amp
from torch.optim import AdamW
from apex.optimizers import FusedAdam
from transformers import get_linear_schedule_with_warmup

from util import *
from dp_engine import DPEngine

SAMPLE_RATE = 1600


def to_audio_index(audio, max_length_value, min_length_value=None):
    # 不是在合法区域内寻找最大值，而是先寻找最大值然后并入合法区域。若输出不合法，则默认为最小长度值
    audio = (audio.argmax(-1).long() * SAMPLE_RATE)
    audio = audio.clamp(min_length_value, max_length_value)
    return audio


class DPTrainer:

    def __init__(self, args, tokenizer):
        self.model, self.config, self.opt, self.sch = self.init_trainable(args, DPEngine, None, args.lr, audio)
        self.args = args
        self.device = args.device
        self.num_ends = self.config.fused.num_ends
        self.hidden_size = self.config.text.hidden_size
        self.mask_token = tokenizer.mask_token
        self.pad_token = tokenizer.pad_token
        self.max_audio_length = 100
        self.temperature = 1
        self.kl_ctl = 0.1
        self.step_count = 0

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

    def train_dp(self, audio_input, audio_mask, text_input, text_mask, mlm_label=None, turn_id=None, valid_filter=None,
                 splits=None, full_text=None):
        """
            audio_input = 3B * 160000; audio_mask = 3B * 160000
            text_input = 2B * 8轮ML; text_mask = 2B * 8轮ML; mlm_label = B * 8轮ML; turn_id=2*ML
            full_text = B * 8轮ML; full_text_mask = B * 8轮ML
            valid_filter = B * 8轮ML; turn_id = 2B * 8轮ML
            splits: B*2大小的列表, 代表第7轮、第8轮文本中词的个数。用splits配合actor_start可分离第7轮和第8轮的词语。
            综上所述: offsets将text_feature切分为history, anchor和positive
                    start_valid和end_valid从text_feature获得anchor和positive的词位置
                    start_valid_for_rm和end_valid_for_rm从词级别RM获取词位置
                    splits将actor_start和actor_end分成两轮
        """
        bs, text_len = mlm_label.shape
        device = mlm_label.device
        # 1 训练MLM&MAM&CRS并优化一次
        mlm, mam, rs_loss = self.model(audio_input, text_input, audio_mask, text_mask, mlm_label=mlm_label,
                                       turn_id=turn_id, valid_filter=valid_filter, stage=1)
        step1_loss = mlm + mam + rs_loss
        self.step_count += 1
        step(step1_loss, self.model, self.args, self.opt, self.sch, self.step_count)
        # 2 no grad时序预测一次
        with torch.no_grad():
            # 2.1 根通过编码器
            audio_input_no_crs = audio_input.view(bs, 3, -1)[:, :2].contiguous().view(bs * 2, -1)
            audio_mask_no_crs = audio_mask.view(bs, 3, -1)[:, :2]
            duration, af_mask = self.model(audio_input_no_crs, full_text, audio_mask_no_crs, full_text_mask,
                                           turn_id=turn_id, valid_filter=valid_filter, stage=2)
            # af_mask表示encoder处理后的audio_attention_mask
            audio_length = audio_mask_no_crs.view(bs, 2, 101).sum(-1).view(-1) - 1  # 减去句首的audio_cls
            text_words = text_input.masked_select(valid_filter).tolist()
            # 2.2将duration按照每轮对话分离
            duration_by_turn, text_words_by_turn = [], []
            for i, s in enumerate(splits):  # B次循环
                duration_by_turn.append(duration[:s[0]] if i == 0 else duration[splits[i - 1][1]:s[0]])
                duration_by_turn.append(duration[s[0]:s[1]])
                text_words_by_turn.append(text_words[:s[0]] if i == 0 else text_words[splits[i - 1][1]:s[0]])
                text_words_by_turn.append(text_words[s[0]:s[1]])
            word_labels = []
            for i, (ml, x) in enumerate(zip(audio_length, duration_by_turn)):  # 2B次循环
                x = (x / self.temperature).softmax(0) * ml
                # 限制每个单词长度至少为1个token，且总长不超过ml
                x_int = x.round().clamp(1, ml)
                x_int, x = x_int.tolist(), x.tolist()
                x_sum = sum(x_int)
                while x_sum != ml:
                    if x_sum > ml:
                        x_max, x_argmax = -1, -1
                        for j, (xi, xs) in enumerate(zip(x_int, x)):
                            if xi == 1:
                                continue
                            elif xi - xs > x_max:
                                x_max, x_argmax = xi - xs, i
                        assert x_argmax >= 0
                        x_int[x_argmax] -= 1
                        x_sum -= 1
                    else:
                        x_min, x_argmin = 1, -1
                        for j, (xi, xs) in enumerate(zip(x_int, x)):
                            if xi - xs < x_min:
                                x_min, x_argmin = xi - xs, i
                        assert x_argmin >= 0
                        x_int[x_argmin] += 1
                        x_sum += 1
                # x_int为每个单词的长度，按照这个整理出音频的label
                word_label = [-100]
                for (w, xi) in zip(text_words_by_turn, x_int):
                    word_label.extend([w] * xi)
                assert len(word_label) == ml + 1
                word_label.extend([-100] * (self.max_audio_length - ml))
                word_labels.append(word_label)
            word_labels = torch.LongTensor(word_labels, device=device)
        # 3 训练语音重构一次（model no grad，只训练重构层）
        full_masked_text = torch.where(full_text == self.pad_token, self.pad_token, self.mask_token)
        step3_loss = self.model(audio_input, full_masked_text, audio_mask_no_crs, text_mask, mlm_label=mlm_label,
                                turn_id=turn_id, valid_filter=valid_filter, stage=3, reconstruct_label=word_labels)
        self.step_count += 1
        step(step3_loss, self.model, self.args, self.opt, self.sch, self.step_count)
        # 4 no grad动态规划一次
        with torch.no_grad():
            raw_predict = self.model(audio_input, full_masked_text, audio_mask_no_crs, text_mask, stage=3,
                                     turn_id=turn_id, mlm_label=mlm_label, valid_filter=valid_filter)  # 2B * 101 * V
            lengths = []
            for i, (ml, ws) in enumerate(zip(audio_length, text_words_by_turn)):
                audio_word_matrix = raw_predict[i, 1:ml + 1, torch.LongTensor(ws, device=device)]  # shape: ML * N
                N = audio_word_matrix.shape[-1]
                F = torch.zeros_like(audio_word_matrix, device=device)
                F[0, :] = float("-inf")
                F[:, 0] = audio_word_matrix[:, 0]
                audio_word_matrix, F = audio_word_matrix.tolist(), F.tolist()
                for j in range(1, ml):
                    for k in range(1, N):
                        F[j][k] = max(F[j - 1][k - 1], F[j - 1][k]) + audio_word_matrix[j][k]
                length = [0] * N
                curr, last = N - 1, ml - 1
                for j in range(ml - 2, 0, -1):
                    if F[j - 1][curr - 1] > F[j - 1][curr]:
                        length[curr] = last - j
                        curr -= 1
                        last = j
                length[0] = last + 1
                assert min(length) >= 1 and sum(length) == ml
                lengths.append(torch.FloatTensor([x / ml for x in length]))
        # 5 前向TPP一次，优化KL散度和对话一致性
        duration, _ = self.model(audio_input_no_crs, full_text, audio_mask_no_crs, text_mask, turn_id=turn_id,
                                 valid_filter=valid_filter, stage=2)
        duration_by_turn = []
        for i, x in enumerate(duration_by_turn):  # B次循环
            duration_by_turn.append(duration[:s[0]] if i == 0 else duration[splits[i - 1][1]:s[0]])
            duration_by_turn.append(duration[s[0]:s[1]])
        duration_by_turn = [(x / self.temperature).softmax(0) for x in duration_by_turn]
        kl_loss, consistent_loss = 0, 0
        for i, (s, l) in enumerate(zip(duration_by_turn, lengths)):
            kl_loss += torch.mean(x * torch.log(x / torch.clamp_min(l.to(dtype=x.dtype, device=device), 1e-4)))
            if i & 4 == 2:
                consistent_loss += torch.mean(x * torch.log(x / torch.clamp_min(duration_by_turn[i - 1], 1e-4)))
        step5_loss = kl_loss / 2 / bs
        if bs > 1:
            step5_loss += consistent_loss / (bs - 1)
        self.step_count += 1
        step(step5_loss, self.model, self.args, self.opt, self.sch, self.step_count)
        return mlm, mam, kl_loss, consistent_loss, rs_loss, step1_loss, step3_loss, step5_loss

    def init_trainable(self, args, model_class, model_name, lr, **model_kwargs):
        # 1 ds config
        if args.ds_config == "default":
            args.ds_config = get_train_ds_config(args.batch_size, torch.cuda.device_count(), args.grad_acc,
                                                 args.ds_stage, args.apex_level)
        elif args.ds_config:
            with open(args.ds_config, "w+") as f:
                args.ds_config = json.load(f)
        # 2 model
        if model_name is None:
            model = model_class(audio=model_kwargs.pop('audio', None), text=model_kwargs.pop('text', None))
        else:
            model = model_class.from_pretrained(model_name, **model_kwargs)
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
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=args.num_train_steps)
        # 4 parallel
        if args.ds_config:
            model, optimizer, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer,
                                                                  config=args.ds_config, lr_scheduler=scheduler,
                                                                  dist_init_required=True)
        else:
            model.to(args.device)
            if args.apex_level > 0:
                model, optimizer = amp.initialize(model, optimizer, opt_level=f"O{args.apex_level}",
                                                  keep_batchnorm_fp32=False if args.apex_level >= 2 else None,
                                                  loss_scale="dynamic" if args.loss_scale == 0. else args.loss_scale)
            if args.local_rank >= 0:
                model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank],
                            output_device=[args.local_rank])
        return model, config, optimizer, scheduler

    def save_pretrained(self, path):
        temp = self.model
        while hasattr(temp, "module"):
            temp = temp.module
        temp.save_pretrained(path)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
