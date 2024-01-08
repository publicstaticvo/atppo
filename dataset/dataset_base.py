import pickle
import numpy as np
from torch.utils.data import Dataset
from util import *


class ATDataset(Dataset):
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


class DataCollatorForAT:
    def __init__(self, tokenizer, config, fp16=False, mlm_prob=0.15):
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
        raise NotImplementedError
