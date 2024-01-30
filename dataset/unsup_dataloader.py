import random

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
        return {
            "audio_input": audios,
            "audio_attention_mask": a_mask,
            "text_input": text,
            "text_attention_mask": t_mask,
            "start_valid": start_valid,
            "split_marks": split_marks,
            "turn_id": token_type
        }


class UnsupervisedDataCollator(DataCollatorForAT):

    def __init__(self, tokenizer, config, fp16=False, mlm_prob=0.15, reconstruct=False):
        super(UnsupervisedDataCollator, self).__init__(tokenizer, config, fp16, mlm_prob)
        self.reconstruct = reconstruct

    def mask_single_word(self, text_input, transcript, mask_idx=None):
        if mask_idx is None:
            word = random.choice(transcript)
        else:
            word = transcript[mask_idx]
        mask_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        labels = text_input.clone()
        start, end = word[1:3]
        labels[:start] = -100
        labels[end:] = -100
        r = random.random()
        if r < 0.8:
            text_input[start:end] = mask_token
        elif r < 0.9:
            random_words = [random.randint(0, len(self.tokenizer) - 3) for _ in range(start, end)]
            text_input[start:end] = torch.LongTensor(random_words)
        return text_input, labels

    def __call__(self, batch):
        return_dict = {
            "audio_input": [],
            "audio_attention_mask": [],
            "text_input": [],
            "text_attention_mask": [],
            "mlm_labels": [],
            "turn_id": [],
            "head_mask_for_fused": []
        }
        ml = 0
        for item in batch:
            ml = max([ml, len(item[1]) + len(item[4]) + len(item[6]) - 2])
        ml = min(ml, self.config.text.max_length)
        seq_length = ml + self.config.audio.max_length * 20 // SAMPLE_RATE
        for item in batch:
            # at和pt 有0有2 history为N-2轮 有0有2 每一轮用2分隔
            aa, at, atr, pa, pt, ptr, history = item
            aa, pa = map(torch.HalfTensor if self.fp16 else torch.FloatTensor, [aa, pa])
            text = torch.LongTensor(history + at[1:] + pt[1:])
            offset_p = (ml - len(pt)) if text.shape[0] > ml else (len(history) - 2 + len(at))
            if self.reconstruct:
                head = torch.zeros((seq_length, seq_length), dtype=torch.bool)
                head[ml:, :ml] = True
                head[:ml, ml:] = True
                return_dict["head_mask_for_fused"].append(head)
            else:
                text, mlm_label = self.get_mlm_instance(text)
                mlm_label, _ = pad_cut(mlm_label, ml, -100)
                return_dict["mlm_labels"].append(mlm_label)
            text, tam = pad_cut(text, ml)
            return_dict["text_input"].append(text)
            return_dict["text_attention_mask"].append(tam)

            aa, a_aam = pad_cut(aa, self.config.audio.max_length)
            pa, p_aam = pad_cut(pa, self.config.audio.max_length)
            return_dict["audio_input"].extend([aa, pa])
            return_dict["audio_attention_mask"].extend([a_aam, p_aam])
            tid = torch.cat([torch.zeros(offset_p + 1), torch.ones(ml - offset_p - 1)]).long()
            return_dict["turn_id"].append(tid)
        return_dict = {k: torch.stack(v, dim=0) for k, v in return_dict.items() if v}
        return return_dict
