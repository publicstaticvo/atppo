import torch
import random
import pickle
import numpy as np
from torch.utils.data import Dataset
from util import pad_cut
SAMPLE_RATE = 16000


def compute_valid(transcript, offset, length, mode, audio_length):
    sv = [0 for _ in range(length)]
    ev = [0 for _ in range(length)]
    start_labels, end_labels = [], []
    for i, item in enumerate(transcript):
        sv[offset + item[-4]] = 1
        ev[offset + item[-3] - 1] = 1
        sl, el = float(f"{item[-2] / audio_length:.3f}"), float(f"{item[-1] / audio_length:.3f}")
        if mode:
            start_labels.append(int(sl * 100))
            end_labels.append(int(el * 100) - 1)
        else:
            start_labels.append(sl)
            end_labels.append(el)
    return torch.BoolTensor(sv), torch.BoolTensor(ev), start_labels, end_labels


class TPPDataset(Dataset):
    def __init__(self, datas, num_turns, file_prefix=None):
        if isinstance(datas, str):
            with open(datas, "rb") as f:
                self.datas = pickle.load(f)
        else:
            self.datas = datas
        self.n = len(datas)
        self.prefix = file_prefix
        self.num_turns = num_turns
        self.has_positive = [i for i, d in enumerate(datas) if d[-1] >= 0]

    def __len__(self):
        return len(self.has_positive)

    def __getitem__(self, idx):
        positive_idx = self.has_positive[idx]  # 0轮
        # print(f"turns: {self.indexs} device: {torch.distributed.get_rank()} idx: {idx} anchor: {anchor_idx}")
        anchor_idx = self.datas[positive_idx][-1]  # -1轮
        negative_idx_audio = random.randint(0, self.n - 3)
        if negative_idx_audio >= positive_idx:
            negative_idx_audio += 2
        negative_idx_text = random.randint(0, self.n - 3)
        if negative_idx_text >= positive_idx:
            negative_idx_text += 2
        history = []  # <-2轮
        curr_idx = anchor_idx
        for i in range(2, self.num_turns):
            if self.datas[curr_idx][-1] == -1:
                break
            curr_idx = self.datas[curr_idx][-1]
            history = self.datas[curr_idx][1][1:] + history
        af, aw = self.datas[anchor_idx][:2]
        at = self.datas[anchor_idx][2:-1]
        pf, pw = self.datas[positive_idx][:2]
        pt = self.datas[positive_idx][2:-1]
        nf = self.datas[negative_idx_audio][0]
        nw = self.datas[negative_idx_text][1]
        if self.prefix is not None:
            af, pf, nf = map(lambda x: x.replace("/mnt/ewwe/yts/at", self.prefix), [af, pf, nf])
        return np.load(af), aw, at, np.load(pf), pw, pt, np.load(nf), nw, [0] + history


class DataCollatorForTPP:
    def __init__(self, tokenizer, config, fp16=False, mlm_prob=0.15):
        self.audio_length = config.audio.max_length
        self.mode = config.fused.num_ends > 1
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
        self.config = config
        self.fp16 = fp16

    def get_mlm_instance(self, text_input):
        # text_input: tokenizer.encode之后的word indices列表。
        labels = text_input.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        # special_tokens_mask：指定序列中哪些位置是special tokens，这些部分不能被mask。主要是[PAD][CLS][SEP]
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 使用labels[masked_indices]作为目标，或直接丢给RobertaForMaskedLM
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        text_input[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        text_input[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return text_input, labels

    def __call__(self, batch):
        audios, a_mask, masked_text, text_labels, t_mask, start_valid, end_valid, token_type, starts, ends = [], [], [], [], [], [], [], [], [], []
        ml = 0
        for item in batch:
            ml = max([ml, len(item[1]) + len(item[4]) + len(item[8]) - 2, len(item[1]) + len(item[7]) + len(item[8]) - 2])
        ml = min(ml, self.config.text.max_length)
        for item in batch:
            # a: -1轮 p: 0轮 n: 负样本 history：<-2轮  一个完整句子由history(8)+at[1:](1)+pt[1:](4)组成
            aa, at, atr, pa, pt, ptr, na, nt, history = item
            # 文本pad之后有两个
            history, at, pt, nt = map(lambda x: torch.LongTensor(x), [history, at, pt, nt])
            ht, h_mlm_label = self.get_mlm_instance(history)
            at, a_mlm_label = self.get_mlm_instance(at[1:])
            pt, p_mlm_label = self.get_mlm_instance(pt[1:])
            nt, _ = self.get_mlm_instance(nt[1:])
            positive = torch.cat([ht, at, pt])
            negative = torch.cat([ht, at, nt])
            if positive.shape[0] > ml:
                offset_p = ml - pt.shape[0] - 1
                offset_a = offset_p - at.shape[0]
            else:
                offset_a = history.shape[0] - 1
                offset_p = offset_a + at.shape[0]
            if negative.shape[0] > ml:
                offset_n = ml - nt.shape[0] - 1
            else:
                offset_n = offset_a + at.shape[0]
            p_text, p_tam = pad_cut(positive, ml)
            n_text, n_tam = pad_cut(negative, ml)
            asv, aev, asl, ael = compute_valid(atr, offset_a, offset_p, self.mode, self.audio_length)
            psv, pev, psl, pel = compute_valid(ptr, 0, ml - offset_p, self.mode, self.audio_length)
            sv = torch.cat([asv, psv])
            ev = torch.cat([aev, pev])
            start_valid.append(sv)
            end_valid.append(ev)
            starts.extend(asl + psl)
            ends.extend(ael + pel)
            p_token_type = torch.cat([torch.zeros(offset_p + 1), torch.ones(ml - offset_p - 1)]).long()
            n_token_type = torch.cat([torch.zeros(offset_n + 1), torch.ones(ml - offset_n - 1)]).long()
            mlm_label, _ = pad_cut(torch.cat([h_mlm_label, a_mlm_label, p_mlm_label]), ml, -100)
            masked_text.extend([p_text, n_text])
            t_mask.extend([p_tam, n_tam])
            text_labels.append(mlm_label)
            token_type.extend([p_token_type, n_token_type])
            # 音频有三个
            aa, pa, na = map(torch.HalfTensor if self.fp16 else torch.FloatTensor, [aa, pa, na])
            aa, a_aam = pad_cut(aa, self.config.audio.max_length)
            pa, p_aam = pad_cut(pa, self.config.audio.max_length)
            na, n_aam = pad_cut(na, self.config.audio.max_length)
            audios.extend([aa, pa, na])
            a_mask.extend([a_aam, p_aam, n_aam])
        audios, a_mask, masked_text, text_labels, t_mask, start_valid, end_valid, token_type = map(
            lambda x: torch.stack(x, dim=0),
            [audios, a_mask, masked_text, text_labels, t_mask, start_valid, end_valid, token_type]
        )
        starts, ends = map(torch.LongTensor if self.mode else (lambda x: torch.tensor(x, dtype=audios.dtype)), [starts, ends])
        return audios, a_mask, masked_text, text_labels, t_mask, start_valid, end_valid, token_type, starts, ends
