from util import *


class DataCollatorForSingleTurnWRM:

    def __init__(self, tokenizer, config, fp16=False):
        self.num_negative = config.num_negative
        self.tokenizer = tokenizer
        self.config = config
        self.fp16 = fp16

    def __call__(self, batch):
        audios, a_mask, texts, t_mask, a_valid, t_valid, negative_indices = [], [], [], [], [], [], []
        ml = 0
        for item in batch:
            ml = max([ml, len(item[1])])
        ml = min(ml, self.config.text.max_length)
        ma = self.config.audio.max_length + 1600  # 161600
        for item in batch:
            a, t, tr = item
            a = torch.HalfTensor(a) if self.fp16 else torch.FloatTensor(a)
            a, am = pad_cut(a, ma)
            t = torch.LongTensor(t)
            t, tm = pad_cut(t, ml)

            text_marks = [[x[1], x[2]] for x in tr]
            audio_marks = [[x[3], x[4]] for x in tr]
            audio_marks = group_scale_audio_length(torch.LongTensor(audio_marks), self.config.audio) + 1  # audio_cls
            audio_valid = compute_valid_for_rm(audio_marks.tolist(), 101, self.config.audio.pooling_mode)
            text_valid = compute_valid_for_rm(text_marks, ml, self.config.text.pooling_mode)
            assert len(audio_valid) == len(tr)
            # 负采样
            negative_indices.append(negative_sampling(tr, self.num_negative))
            audios.append(a)
            a_mask.append(am)
            texts.append(t)
            t_mask.append(tm)
            a_valid.append(audio_valid)
            t_valid.append(text_valid)
        audios, a_mask, texts, t_mask = map(lambda x: torch.stack(x, dim=0), [audios, a_mask, texts, t_mask])
        return {
            "audio_input": audios,
            "audio_mask": a_mask,
            "text_input": texts,
            "text_mask": t_mask,
            "audio_valid": a_valid,
            "text_valid": t_valid,
            "neg": negative_indices
        }
