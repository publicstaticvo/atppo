import torch
import pickle
from torch.utils.data import Dataset


def pad_cut(sequence, length):
    seq_len = sequence.shape[0]
    if length > seq_len:
        padding = torch.zeros(length - seq_len, dtype=sequence.dtype)
        att = torch.cat([torch.ones_like(sequence), padding])
        sequence = torch.cat([sequence, padding])
    else:
        if sequence.dtype == torch.long:
            # sequence = torch.cat([sequence[:1], sequence[1 - length:]])
            sequence = torch.cat([sequence[:length - 1], sequence[-1:]])
        else:
            sequence = sequence[:length]
        att = torch.ones_like(sequence)
    return sequence, att.to(dtype=torch.long)


class DownstreamDataset(Dataset):
    def __init__(self, root, task, op):
        sub_root = "downstreamv2"
        if task in ["meld", "iemocap"]:
            audio_multi_turn = True
            task += "2"
        else:
            audio_multi_turn = False
        if op == "train":
            print(sub_root, task)
        with open(f"{root}/{sub_root}/{task}/{op}.pkl", "rb") as f:
            self.data_list = pickle.load(f)
        if audio_multi_turn:
            for i, item in enumerate(self.data_list[1]):
                if item[3] >= 0:
                    word = item[4] + item[1][1:]
                    turn_id = [0 for _ in item[4]] + [1 for _ in range(len(word) - len(item[4]))]
                    audio = self.data_list[0][item[3]]
                else:
                    word = item[1]
                    turn_id = [1 for _ in item[1]]
                    audio = []
                self.data_list[1][i] = [self.data_list[0][item[0]], word, item[2], turn_id, audio]
            self.data_list = self.data_list[1]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


class DataCollatorForDownstream:
    def __init__(self, audio_length, float_label, min_text_length):
        self.audio_length = audio_length
        self.float_label = float_label
        self.min_text_length = min_text_length

    def __call__(self, batch):
        audios, a_mask, texts, labels, t_mask, turn_ids = [], [], [], [], [], []
        ml = self.min_text_length
        for item in batch:
            ml = max(ml, len(item[1]))
        ml = min(ml, 512)
        for item in batch:
            audio, text, label = item[:3]
            text, tam = pad_cut(torch.LongTensor(text), ml)
            texts.append(text)
            t_mask.append(tam)
            labels.append(label)
            if len(item) > 4:
                prev_audio, pam = pad_cut(torch.HalfTensor(item[4]), self.audio_length)
                audios.append(prev_audio)
                a_mask.append(pam)
            audio, aam = pad_cut(torch.HalfTensor(audio), self.audio_length)
            audios.append(audio)
            a_mask.append(aam)
            if len(item) > 3:
                token_type = pad_cut(torch.LongTensor(item[3]), ml)[0]
                turn_ids.append(token_type)
        audios, a_mask, texts, t_mask = map(lambda x: torch.stack(x, dim=0), [audios, a_mask, texts, t_mask])
        return_dict = {"audio": audios, "text": texts, "aam": a_mask, "tam": t_mask,
                       "label": torch.HalfTensor(labels) if self.float_label else torch.LongTensor(labels),
                       "turn_id": torch.stack(turn_ids, dim=0) if turn_ids else None}
        return return_dict
