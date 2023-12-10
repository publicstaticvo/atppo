import json
import torch
import pickle
import random
import numpy as np
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
    def __init__(self, root, task, op, mode, tokenizer, v2=False, prompt=False, audio_multi_turn=False):
        if mode:
            try:
                filename = f"{root}/{task}/{op}_{mode}_{int(prompt)}.pkl" if audio_multi_turn else f"{root}/{task}/{op}_{mode}_{int(prompt)}v1.pkl"
                with open(filename, "rb") as f:
                    self.data_list = pickle.load(f)
            except:
                with open(f"{root}/{task}/{op}_{mode}.json") as f:
                    self.data_list = json.load(f)
                for i in range(len(self.data_list)):
                    text = self.data_list[i][4] + [self.data_list[i][1]]
                    turn = [0 for _ in self.data_list[i][4]] + [1]
                    if prompt:
                        speaker = self.data_list[i][1][:self.data_list[i][1].find(":")]
                        text.append(f"{speaker} feels <mask>.")
                        turn.append(1)
                    if v2:
                        temp = [0]
                        token_type = [0 if self.data_list[i][4] else 1]
                        for j, sentence in enumerate(text):
                            for word in sentence.split(" "):
                                w = tokenizer.encode(word)[1:-1]
                                temp.extend(w)
                                token_type.extend([turn[j] for _ in w])
                            temp.append(2)
                            token_type.append(turn[j])
                    else:
                        temp = [0]
                        token_type = [0 if self.data_list[i][4] else 1]
                        for j, sentence in enumerate(text):
                            w = tokenizer.encode(sentence)[1:]
                            temp.extend(w)
                            token_type.extend([turn[j] for _ in w])
                    if audio_multi_turn:
                        if self.data_list[i][3] and self.data_list[i][3][-1]:
                            self.data_list[i] = [np.load(f"{root}/{task}/raw/{self.data_list[i][0]}"), temp,
                                                 self.data_list[i][2],
                                                 token_type, np.load(f"{root}/{task}/raw/{self.data_list[i][3][-1]}")]
                        else:
                            self.data_list[i] = [np.load(f"{root}/{task}/raw/{self.data_list[i][0]}"), temp,
                                                 self.data_list[i][2], token_type, []]
                    else:
                        self.data_list[i] = [np.load(f"{root}/{task}/raw/{self.data_list[i][0]}"), temp,
                                             self.data_list[i][2], token_type]
                filename = f"{root}/{task}/{op}_{mode}_{int(prompt)}.pkl" if audio_multi_turn else f"{root}/{task}/{op}_{mode}_{int(prompt)}v1.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(self.data_list, f)
        else:
            sub_root = "downstreamv2" if v2 else "downstream"
            if prompt:
                task += "p"
            elif audio_multi_turn:
                task += "2"
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
            elif prompt:
                for i, item in enumerate(self.data_list[1]):
                    self.data_list[1][i] = [self.data_list[0][item[0]]] + item[1:3]
                self.data_list = self.data_list[1]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


class DataCollatorForDownstream:
    def __init__(self, audio_length, float_label, min_text_length, prompt=False):
        self.audio_length = audio_length
        self.float_label = float_label
        self.prompt = prompt
        self.min_text_length = min_text_length

    def __call__(self, batch):
        audios, a_mask, texts, labels, t_mask, turn_ids, prompts = [], [], [], [], [], [], []
        ml = self.min_text_length
        for item in batch:
            ml = max(ml, len(item[1]))
        ml = min(ml, 512)
        for item in batch:
            audio, text, label = item[:3]
            if self.prompt:
                prompts.append(min(len(text), ml) - 3)
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
        audios, a_mask, texts, t_mask = map(
            lambda x: torch.stack(x, dim=0),
            [audios, a_mask, texts, t_mask]
        )
        return_dict = {"audio": audios, "text": texts, "aam": a_mask, "tam": t_mask,
                       "label": torch.HalfTensor(labels) if self.float_label else torch.LongTensor(labels),
                       "turn_id": torch.stack(turn_ids, dim=0) if turn_ids else None,
                       "prompt": torch.LongTensor(prompts) if prompts else None}
        return return_dict
