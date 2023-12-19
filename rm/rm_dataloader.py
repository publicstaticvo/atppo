import numpy as np
from torch.utils.data import Dataset
from util import *


class RMDataset(Dataset):
    def __init__(self, datas, num_turns, file_prefix=None):
        if isinstance(datas, str):
            with open(datas, "rb") as f:
                self.datas = pickle.load(f)
        else:
            self.datas = datas
        self.n = len(datas)
        self.prefix = file_prefix
        self.num_turns = num_turns
        self.has_positive = [i for i, d in enumerate(self.datas) if d[-1] >= 0]

    def __len__(self):
        return len(self.has_positive)

    def __getitem__(self, idx):
        positive_idx = self.has_positive[idx]  # 0轮
        anchor_idx = self.datas[positive_idx][-1]  # -1轮
        history = []  # <-2轮
        curr_idx = anchor_idx
        for i in range(2, self.num_turns):
            if self.datas[curr_idx][-1] == -1:
                break
            curr_idx = self.datas[curr_idx][-1]
            history = self.datas[curr_idx][1][1:] + history
        pf, pw = self.datas[positive_idx][:2]
        pt = self.datas[positive_idx][2:-1]
        af, aw = self.datas[anchor_idx][:2]
        at = self.datas[anchor_idx][2:-1]
        if self.prefix is not None:
            pf, af = map(lambda x: x.replace("/mnt/ewwe/yts/at", self.prefix), [pf, af])
        return np.load(af), aw, at, np.load(pf), pw, pt, [0] + history


class DataCollatorForRM:
    def __init__(self, tokenizer, config, fp16=False, mlm_prob=0.15):
        self.num_negative = config.num_negative
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
        audios, a_mask, texts, t_mask, a_valid, t_valid, turn_id, labels, negative_indices = [], [], [], [], [], [], [], [], []
        ml = 0
        for item in batch:
            ml = max([ml, len(item[1]) + len(item[4]) + len(item[6]) - 2])
        ml = min(ml, self.config.text.max_length)
        for item in batch:
            # at和pt 有0有2 history为N-2轮 有0有2 每一轮用2分隔
            aa, at, atr, pa, pt, ptr, history = item
            aa, pa = map(torch.HalfTensor if self.fp16 else torch.FloatTensor, [aa, pa])
            aa, a_aam = pad_cut(aa, self.config.audio.max_length)
            pa, p_aam = pad_cut(pa, self.config.audio.max_length)
            text = torch.LongTensor(history + at[1:] + pt[1:])
            if text.shape[0] > ml:
                offset_p = ml - len(pt)
                offset_a = offset_p - len(at) + 1
            else:
                offset_a = len(history) - 1
                offset_p = offset_a + len(at) - 1
            text, tam = pad_cut(text, ml)
            if self.config.perform_mlm:
                text, mlm_label = self.get_mlm_instance(text)
                labels.append(mlm_label)

            text_marks = []
            anchor_audio_marks = []
            positive_audio_marks = []
            for i, x in enumerate(atr):
                anchor_audio_marks.append([x[3], x[4]])
                text_marks.append([x[1] + offset_a, x[2] + offset_a])
            for i, x in enumerate(ptr):
                positive_audio_marks.append([x[3], x[4]])
                text_marks.append([x[1] + offset_p, x[2] + offset_p])

            anchor_audio_marks = group_scale_audio_length(torch.LongTensor(anchor_audio_marks), self.config.audio)
            positive_audio_marks = group_scale_audio_length(torch.LongTensor(positive_audio_marks), self.config.audio)
            audio_length = scale_audio_length(0, self.config.audio.max_length, self.config.audio)
            anchor_audio_marks -= audio_length[0]
            positive_audio_marks -= audio_length[0]
            for i in range(anchor_audio_marks.shape[0] - 1, -1, -1):
                if anchor_audio_marks[i, 1] > audio_length[1]: anchor_audio_marks[i, 1] -= 1
                else: break
                if anchor_audio_marks[i, 0] == anchor_audio_marks[i, 1]: anchor_audio_marks[i, 0] -= 1
            for i in range(positive_audio_marks.shape[0] - 1, -1, -1):
                if positive_audio_marks[i, 1] > audio_length[1]: positive_audio_marks[i, 1] -= 1
                else: break
                if positive_audio_marks[i, 0] == positive_audio_marks[i, 1]: positive_audio_marks[i, 0] -= 1
            anchor_audio_marks += 1
            positive_audio_marks += (anchor_audio_marks[-1, 1] + 1)
            # 0在scale_audio_length之后会变-1
            audio_marks = torch.cat([anchor_audio_marks, positive_audio_marks], dim=0)
            audio_valid, _ = compute_valid(audio_marks.tolist(), audio_length[1] * 2 + 2, self.config.audio.pooling_mode)
            text_valid, num = compute_valid(text_marks, ml, self.config.text.pooling_mode)
            assert num == len(audio_valid) == len(atr) + len(ptr)
            # 负采样
            negative_indices.append(negative_sampling(atr + ptr, self.num_negative))

            text, tam = pad_cut(text, ml)
            audios.extend([aa, pa])
            a_mask.extend([a_aam, p_aam])
            texts.append(text)
            t_mask.append(tam)
            a_valid.append(audio_valid)
            t_valid.append(text_valid)
            turn_id.append(torch.cat([torch.zeros(offset_p + 1), torch.ones(ml - offset_p - 1)]).long())
        audios, a_mask, texts, t_mask, turn_id = map(lambda t: torch.stack(t, dim=0), [audios, a_mask, texts, t_mask, turn_id])
        if labels: labels = torch.stack(labels, dim=0)
        else: labels = None
        return audios, a_mask, a_valid, texts, t_mask, t_valid, turn_id, negative_indices, labels
