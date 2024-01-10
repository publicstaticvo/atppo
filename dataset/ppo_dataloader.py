import torch
from .dataset_base import DataCollatorForAT
from util import pad_cut, compute_valid_for_tpp
SAMPLE_RATE = 16000


class DataCollatorForPPO(DataCollatorForAT):
    def __init__(self, args, tokenizer, mlm_prob=0.15):
        super(DataCollatorForPPO, self).__init__(tokenizer, args, fp16=args.apex_level > 0, mlm_prob=mlm_prob)
        self.audio_length = int(args.audio_length * SAMPLE_RATE)
        self.mode = args.num_ends > 1

    def __call__(self, batch):
        audios, a_mask, text, masked_text, text_labels, t_mask, start_valid, end_valid, valid_filter, token_type, split_marks = [], [], [], [], [], [], [], [], [], [], []
        ml = 0
        for item in batch:
            ml = max([ml, len(item[1]) + len(item[4]) + len(item[8]) - 2, len(item[1]) + len(item[7]) + len(item[8]) - 2])
        ml = min(ml, self.config.max_length)
        for item in batch:
            # a: -1轮 p: 0轮 n: 负样本 history：<-2轮  一个完整句子由history(8)+at[1:](1)+pt[1:](4)组成
            aa, at, atr, pa, pt, ptr, na, nt, history = item
            # 文本pad之后有两个
            history, at, pt, nt = map(lambda x: torch.LongTensor(x), [history, at, pt, nt])
            t = torch.cat([history, at[1:], pt[1:]])
            t, _ = pad_cut(t, ml)
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
            asv, aev, asl, _ = compute_valid_for_tpp(atr, offset_a, offset_p, self.mode, self.audio_length)
            psv, pev, psl, _ = compute_valid_for_tpp(ptr, 0, ml - offset_p, self.mode, self.audio_length)
            start_valid.extend([asv, psv])
            end_valid.extend([aev, pev])
            split_marks.append(len(asl))
            text.append(t)
            p_token_type = torch.cat([torch.zeros(offset_p + 1), torch.ones(ml - offset_p - 1)]).long()
            n_token_type = torch.cat([torch.zeros(offset_n + 1), torch.ones(ml - offset_n - 1)]).long()
            mlm_label, _ = pad_cut(torch.cat([h_mlm_label, a_mlm_label, p_mlm_label]), ml, -100)
            masked_text.extend([p_text, n_text])
            t_mask.extend([p_tam, n_tam])
            text_labels.append(mlm_label)
            token_type.extend([p_token_type, n_token_type])
            # 音频有三个
            aa, pa, na = map(torch.HalfTensor if self.fp16 else torch.FloatTensor, [aa, pa, na])
            aa, a_aam = pad_cut(aa, self.audio_length)
            pa, p_aam = pad_cut(pa, self.audio_length)
            na, n_aam = pad_cut(na, self.audio_length)
            audios.extend([aa, pa, na])
            a_mask.extend([a_aam, p_aam, n_aam])
        audios, a_mask, text, masked_text, text_labels, t_mask, token_type = map(
            lambda x: torch.stack(x, dim=0),
            [audios, a_mask, text, masked_text, text_labels, t_mask, token_type]
        )
        return audios, a_mask, text, masked_text, text_labels, t_mask, start_valid, end_valid, token_type, split_marks
