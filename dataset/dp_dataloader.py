import torch
from .dataset_base import DataCollatorForAT
from util import pad_cut, compute_valid_for_tpp
SAMPLE_RATE = 16000


class DataCollatorForDP(DataCollatorForAT):
    def __init__(self, args, tokenizer, mlm_prob=0.15):
        super(DataCollatorForDP, self).__init__(tokenizer, args, fp16=args.apex_level > 0, mlm_prob=mlm_prob)
        self.audio_length = int(args.audio_length * SAMPLE_RATE)
        self.mode = args.num_ends > 1

    def __call__(self, batch):
        audios, a_mask, text, t_mask, start_valid, token_type, split_marks = [], [], [], [], [], [], []
        ml = 0
        for item in batch:
            ml = max(ml, len(item[1]) + len(item[4]) + len(item[6]) - 2)
        ml = min(ml, self.config.max_length)
        for item in batch:
            aa, at, atr, pa, pt, ptr, history = item
            history, at, pt = map(lambda x: torch.LongTensor(x), [history, at, pt])
            t = torch.cat([history, at[1:], pt[1:]])
            t, tm = pad_cut(t, ml)
            if t.shape[0] > ml:
                offset_p = ml - pt.shape[0] - 1
                offset_a = offset_p - at.shape[0]
            else:
                offset_a = history.shape[0] - 1
                offset_p = offset_a + at.shape[0]
            asv, _, asl, _ = compute_valid_for_tpp(atr, offset_a, offset_p, self.audio_length)
            psv, *_ = compute_valid_for_tpp(ptr, 0, ml - offset_p, self.audio_length)
            start_valid.extend([asv, psv])
            split_marks.append(len(asl))
            text.append(t)
            t_mask.append(torch.LongTensor(tm))
            token_type.append(torch.cat([torch.zeros(offset_p + 1), torch.ones(ml - offset_p - 1)]).long())
            aa, pa = map(torch.HalfTensor if self.fp16 else torch.FloatTensor, [aa, pa])
            aa, a_aam = pad_cut(aa, self.audio_length)
            pa, p_aam = pad_cut(pa, self.audio_length)
            audios.extend([aa, pa])
            a_mask.extend([a_aam, p_aam])
        audios, a_mask, text, t_mask, token_type = map(lambda x: torch.stack(x, dim=0), [audios, a_mask, text, t_mask, token_type])
        return audios, a_mask, text, t_mask, start_valid, token_type, split_marks
