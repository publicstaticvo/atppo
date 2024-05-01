import torch
import random
import numpy as np
from util import pad_cut, compute_valid_for_tpp
from dataset import ATDataset, DataCollatorForAT

SAMPLE_RATE = 16000


class DPDataset(ATDataset):
    def __init__(self, datas, num_turns, file_prefix=None):
        super(DPDataset, self).__init__(datas, num_turns, file_prefix, three_turns=True)

    def get_item(self, idx):
        # 涉及到对话一致性约束，取样本逻辑变化，一定会取到相邻的两段对话
        positive_idx = self.has_positive[idx]  # 0轮
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

    def __getitem__(self, idx):
        return [self.get_item(idx), self.get_item(self.datas[self.has_positive[idx]][-1])]


class DataCollatorForDP(DataCollatorForAT):
    def __init__(self, args, tokenizer, mlm_prob=0.15):
        super(DataCollatorForDP, self).__init__(tokenizer, args, fp16=args.apex_level > 0, mlm_prob=mlm_prob)
        self.audio_length = int(args.audio_length * SAMPLE_RATE)
        self.text_length = args.max_length

    def __call__(self, batch):
        batch = [y for x in batch for y in x]
        audios, a_mask = [], []
        masked_text, t_mask, full_text, text_labels = [], [], [], []
        valid_filter, splits, turn_ids = [], [], []
        ml = 0
        for item in batch:
            ml = max([ml, len(item[1]) + len(item[4]) + len(item[8]), len(item[1]) + len(item[7]) + len(item[8])])
        ml = min(ml - 2, self.text_length)
        total_words = 0
        for item in batch:
            aa, at, atr, pa, pt, ptr, na, nt, history = item
            # 语音
            aa, pa, na = map(torch.HalfTensor if self.fp16 else torch.FloatTensor, [aa, pa, na])
            aa, a_aam = pad_cut(aa, self.audio_length + 1600)
            pa, p_aam = pad_cut(pa, self.audio_length + 1600)
            na, n_aam = pad_cut(na, self.audio_length + 1600)
            audios.extend([aa, pa, na])
            a_mask.extend([a_aam, p_aam, n_aam])
            # full_text文本
            history, at, pt, nt = map(lambda x: torch.LongTensor(x), [history, at, pt, nt])
            full_text.append(torch.cat([history, at[1:], pt[1:]]))
            # masked文本
            ht, h_mlm_label = self.get_mlm_instance(history)
            at, a_mlm_label = self.get_mlm_instance(at[1:])
            pt, p_mlm_label = self.get_mlm_instance(pt[1:])
            nt, _ = self.get_mlm_instance(nt[1:])
            mlm_label, _ = pad_cut(torch.cat([h_mlm_label, a_mlm_label, p_mlm_label]), ml, -100)
            text_labels.append(mlm_label)
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
                offset_n = history.shape[0] - 1 + at.shape[0]
            # offset_a: 前6轮的长度，无尾2 offset_p: 前7轮的长度，无尾2
            p_text, p_tam = pad_cut(positive, ml)
            n_text, n_tam = pad_cut(negative, ml)
            masked_text.extend([p_text, n_text])
            t_mask.extend([p_tam, n_tam])
            # valid
            asv, _, asl, _ = compute_valid_for_tpp(atr, offset_a, offset_p, self.audio_length)
            psv, _, psl, _ = compute_valid_for_tpp(ptr, 0, ml - offset_p, self.audio_length)
            valid_filter.append(torch.cat([asv, psv]))
            total_words += len(asl)
            splits.append([total_words])
            total_words += len(psl)
            splits[-1].append(total_words)
            # turn_id
            p_token_type = torch.cat([torch.zeros(offset_p + 1), torch.ones(ml - offset_p - 1)]).long()
            n_token_type = torch.cat([torch.zeros(offset_n + 1), torch.ones(ml - offset_n - 1)]).long()
            turn_ids.extend([p_token_type, n_token_type])
        audios, a_mask, masked_text, t_mask, text_labels, valid_filter, full_text, turn_ids = map(
            lambda x: torch.stack(x, dim=0),
            [audios, a_mask, masked_text, t_mask, text_labels, valid_filter, full_text, turn_ids]
        )
        return {
            "audio_input": audios,
            "audio_mask": a_mask,
            "text_input": masked_text,
            "text_mask": t_mask,
            "mlm_label": text_labels,
            "turn_id": turn_ids,
            "valid_filter": valid_filter,
            "full_text": full_text,
            "splits": splits
        }
