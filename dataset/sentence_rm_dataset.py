from util import *
from .dataset_base import DataCollatorForAT


class DataCollatorForSingleTurnSRM:

    def __init__(self, tokenizer, config, fp16=False):
        self.num_negative = config.num_negative
        self.tokenizer = tokenizer
        self.config = config
        self.fp16 = fp16

    def __call__(self, batch):
        audios, a_mask, texts, t_mask = [], [], [], []
        ml = 0
        for item in batch:
            ml = max([ml, len(item[1])])
        ml = min(ml, self.config.text.max_length)
        # ma = 0
        # for item in batch:
        #     ma = max([ma, len(item[2])])
        # ma = self.config.audio.max_length + 1600 * ma
        ma = 2 * self.config.audio.max_length
        for item in batch:
            a, t, tr = item
            a = torch.HalfTensor(a) if self.fp16 else torch.FloatTensor(a)
            ba = construct_audio_batch(a, tr, self.num_negative)
            for b in ba:
                b, bm = pad_cut(b, ma)
                audios.append(b)
                a_mask.append(bm)
            t = torch.LongTensor(t)
            t, tm = pad_cut(t, ml)
            texts.append(t)
            t_mask.append(tm)
        audios, a_mask, texts, t_mask = map(lambda x: torch.stack(x, dim=0), [audios, a_mask, texts, t_mask])
        return {
            "audio_input": audios,
            "audio_mask": a_mask,
            "text_input": texts,
            "text_mask": t_mask
        }
