import numpy as np
from util import *
from .dataset_base import ATDataset, DataCollatorForAT


class SentenceAlignDataset(ATDataset):

    def __init__(self, datas, num_turns, file_prefix=None):
        super(SentenceAlignDataset, self).__init__(datas, num_turns, file_prefix)

    def negative_sampling_step(self, audio, start, end):
        if audio.shape[0] > 81600 and random.random() < 0.7:
            new_start = random.randint(0, audio.shape[0] - end + start - 1600)
            if start <= new_start <= start + 1600:
                new_start += 1600
            new_end = new_start + end - start
            new_audio = torch.cat([audio[:start], audio[new_start:new_end], audio[end:]])
            align_score = min(1, abs(new_start - start) / 10000)
            return new_audio, align_score
        replace_audio = torch.from_numpy(np.load(random.choice(self.datas)[0])).to(dtype=audio.dtype)
        while replace_audio.shape[0] < end - start + 40000:
            replace_audio = torch.from_numpy(np.load(random.choice(self.datas)[0])).to(dtype=audio.dtype)
        new_start = random.randint(0, replace_audio.shape[0] - end + start)
        new_end = new_start + end - start
        new_audio = torch.cat([audio[:start], replace_audio[new_start:new_end], audio[end:]])
        return new_audio, 1

    def negative_sampling(self, history, query, history_transcript, query_transcript):
        num_steps = random.randint(1, len(history_transcript) + len(query_transcript))
        num_history_steps = max(min(num_steps * len(history) // (len(history) + len(query)), len(history_transcript)), num_steps - len(query_transcript))
        num_query_steps = num_steps - num_history_steps
        total_score = 0
        for tr in random.sample(history_transcript, num_history_steps):
            history, score = self.negative_sampling_step(history, tr[-2], tr[-1])
            total_score += score
        for tr in random.sample(query_transcript, num_query_steps):
            query, score = self.negative_sampling_step(query, tr[-2], tr[-1])
            total_score += score
        return history, query, total_score


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
                na, npa, score = self.dataset.negative_sampling(aa, pa, atr, ptr)
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
