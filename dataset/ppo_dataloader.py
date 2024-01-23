import torch
from .dataset_base import DataCollatorForAT
from util import pad_cut, compute_valid_for_tpp

SAMPLE_RATE = 16000


class DataCollatorForPPO(DataCollatorForAT):
    def __init__(self, args, tokenizer, mlm_prob=0.15):
        super(DataCollatorForPPO, self).__init__(tokenizer, args, fp16=args.apex_level > 0, mlm_prob=mlm_prob)
        self.audio_length = int(args.audio_length * SAMPLE_RATE)
        self.text_length = args.max_length

    def __call__(self, batch):
        audios, a_mask = [], []
        masked_text, t_mask, text_labels = [], [], []
        start_valid, end_valid, offsets, splits = [], [], [], []
        text_rm, t_mask_rm, start_valid_rm = [], [], []
        turn_ids = []
        ml = 0
        for item in batch:
            ml = max([ml, len(item[1]) + len(item[4]) + len(item[8]) - 2, len(item[1]) + len(item[7]) + len(item[8]) - 2])
        ml = min(ml, self.text_length)
        ml_for_rm = 0
        for item in batch:
            ml_for_rm = max([ml_for_rm, len(item[1]), len(item[4])])
        ml_for_rm = min(ml_for_rm, self.text_length)
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
            (at_for_rm, at_for_rm_mask), (pt_for_rm, pt_for_rm_mask) = map(lambda x: pad_cut(x, ml_for_rm), [at, pt])
            text_rm.extend([at_for_rm, pt_for_rm])
            t_mask_rm.extend([at_for_rm_mask, pt_for_rm_mask])
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
            real_length = positive.shape[0]
            # offset_a: 前6轮的长度，无尾2 offset_p: 前7轮的长度，无尾2
            p_text, p_tam = pad_cut(positive, ml)
            n_text, n_tam = pad_cut(negative, ml)
            masked_text.extend([p_text, n_text])
            t_mask.extend([p_tam, n_tam])
            # valid
            asv, aev, asl, _ = compute_valid_for_tpp(atr, offset_a, offset_p, self.audio_length)
            psv, pev, psl, _ = compute_valid_for_tpp(ptr, 0, ml - offset_p, self.audio_length)
            start_valid.append(torch.cat([asv, psv]))
            end_valid.append(torch.cat([aev, pev]))
            asvrm = torch.cat([asv[offset_a:], torch.BoolTensor([0])])
            psvrm = psv[:real_length - offset_p]
            assert asvrm.shape[0] == at.shape[0] + 1, (asvrm.shape, at.shape)
            assert psvrm.shape[0] == pt.shape[0] + 1, (psvrm.shape, pt.shape)
            asvrm, psvrm = map(lambda x: pad_cut(x, ml_for_rm)[0], [asvrm, psvrm])
            start_valid_rm.extend([asvrm, psvrm])
            total_words += len(asl)
            splits.append([total_words])
            total_words += len(psl)
            splits[-1].append(total_words)
            # turn_id
            p_token_type = torch.cat([torch.zeros(offset_p + 1), torch.ones(ml - offset_p - 1)]).long()
            n_token_type = torch.cat([torch.zeros(offset_n + 1), torch.ones(ml - offset_n - 1)]).long()
            turn_ids.extend([p_token_type, n_token_type])
        audios, a_mask, masked_text, t_mask, text_labels, start_valid, end_valid, text_rm, t_mask_rm, turn_ids, start_valid_rm = map(
            lambda x: torch.stack(x, dim=0),
            [audios, a_mask, masked_text, t_mask, text_labels, start_valid, end_valid, text_rm, t_mask_rm, turn_ids, start_valid_rm]
        )
        return {
            "audio_input": audios,
            "audio_mask": a_mask,
            "text_input": masked_text,
            "text_mask": t_mask,
            "mlm_label": text_labels,
            "turn_id": turn_ids,
            "start_valid": start_valid,
            "end_valid": end_valid,
            "full_text_for_rm": text_rm,
            "full_text_for_rm_mask": t_mask_rm,
            "start_valid_for_rm": start_valid_rm,
            "splits": splits
        }
