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
        return audios, a_mask, texts, t_mask


class DataCollatorForSentenceRM(DataCollatorForAT):

    def __init__(self, tokenizer, config, dataset, fp16=False, mlm_prob=0.15):
        super(DataCollatorForSentenceRM, self).__init__(tokenizer, config, fp16, mlm_prob)
        self.num_negative = config.num_negative
        self.dataset = dataset

    def __call__(self, batch):
        audios, a_mask, texts, t_mask, turn_id = [], [], [], [], []
        ml = 0
        for item in batch:
            ml = max([ml, len(item[1]) + len(item[4]) + len(item[6]) - 2])
        ml = min(ml, self.config.text.max_length)
        for item in batch:
            # at和pt 有0有2 history为N-2轮 有0有2 每一轮用2分隔
            aa, at, atr, pa, pt, ptr, history = item
            aa, pa = map(torch.HalfTensor if self.fp16 else torch.FloatTensor, [aa, pa])
            text = torch.LongTensor(history + at[1:] + pt[1:])
            offset_p = (ml - len(pt)) if text.shape[0] > ml else (len(history) - 2 + len(at))
            text, tam = pad_cut(text, ml)

            negative_indices = []
            for i in range(self.num_negative):
                na, npa, score = negative_audio(self.dataset, aa, pa, atr, ptr)
                negative_indices.append([na, npa, score])
            negative_indices = sorted(negative_indices, key=lambda x: x[-1])

            aa, a_aam = pad_cut(aa, self.config.audio.max_length)
            pa, p_aam = pad_cut(pa, self.config.audio.max_length)
            audios.extend([aa, pa])
            a_mask.extend([a_aam, p_aam])
            for a, p, _ in negative_indices:
                a, am = pad_cut(a, self.config.audio.max_length)
                p, pm = pad_cut(p, self.config.audio.max_length)
                audios.extend([a, p])
                a_mask.extend([am, pm])
            texts.append(text)
            t_mask.append(tam)
            tid = torch.cat([torch.zeros(offset_p + 1), torch.ones(ml - offset_p - 1)]).long()
            turn_id.append(tid)
        audios, a_mask, texts, t_mask, turn_id = map(lambda t: torch.stack(t, dim=0), [audios, a_mask, texts, t_mask, turn_id])
        return audios, a_mask, texts, t_mask, turn_id
