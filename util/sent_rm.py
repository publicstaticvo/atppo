import torch
import random
import numpy as np


def construct_audio_batch(audio, transcript):
    num_steps = random.randint(1, len(transcript))
    positive = []
    for i, t in enumerate(transcript):
        if i > 0:
            positive.append(torch.zeros(1600))
        positive.append(audio[t[-2]:t[-1]])
    positive = torch.cat(positive)
    negatives = []
    for i in range(num_steps):
        tr = [[t[-2], t[-1]] for t in transcript]
        negative = []
        # 变换坐标点
        steps = random.randint(1, len(transcript))
        for j in random.sample(list(range(len(transcript))), steps):
            tr[j][0] = random.randint(0, 99) * 1600
            tr[j][1] = random.randint(tr[j][0] + 1, 100) * 1600
        for j, t in enumerate(tr):
            if j > 0:
                negative.append(torch.zeros(1600))
            negative.append(audio[t[0]:t[1]])
        negative = torch.cat(negative)
        negatives.append([negative, steps])
    negatives = [x[0] for x in sorted(negatives, key=lambda y: y[1])]
    return [positive] + negatives


def replace_word(datas, audio, start, end):
    if audio.shape[0] > 81600 and random.random() < 0.7:
        new_start = random.randint(0, audio.shape[0] - end + start - 1600)
        if start <= new_start <= start + 1600:
            new_start += 1600
        new_end = new_start + end - start
        new_audio = torch.cat([audio[:start], audio[new_start:new_end], audio[end:]])
        align_score = min(1, abs(new_start - start) / 10000)
        return new_audio, align_score
    replace_audio = torch.from_numpy(np.load(random.choice(datas)[0])).to(dtype=audio.dtype)
    while replace_audio.shape[0] < end - start + 40000:
        replace_audio = torch.from_numpy(np.load(random.choice(datas)[0])).to(dtype=audio.dtype)
    new_start = random.randint(0, replace_audio.shape[0] - end + start)
    new_end = new_start + end - start
    new_audio = torch.cat([audio[:start], replace_audio[new_start:new_end], audio[end:]])
    return new_audio, 1


def negative_audio(data, history, query, history_transcript, query_transcript):
    num_steps = random.randint(1, len(history_transcript) + len(query_transcript))
    num_history_steps = max(min(num_steps * len(history) // (len(history) + len(query)), len(history_transcript)),
                            num_steps - len(query_transcript))
    num_query_steps = num_steps - num_history_steps
    total_score = 0
    for tr in random.sample(history_transcript, num_history_steps):
        history, score = replace_word(data, history, tr[-2], tr[-1])
        total_score += score
    for tr in random.sample(query_transcript, num_query_steps):
        query, score = replace_word(data, query, tr[-2], tr[-1])
        total_score += score
    return history, query, total_score
