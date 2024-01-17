import torch
import random
from torch.nn.functional import normalize


def negative_sampling(words, num_negative):
    words = [x[0][:-1] if x[0][-1] in [',', '.', '?', '!'] and len(x[0]) > 1 else x[0] for x in words]
    negative_samples = torch.zeros([len(words), len(words)], dtype=torch.bool)
    for i, x in enumerate(words):
        idx = [j for j, y in enumerate(words) if y != x]
        if 0 < num_negative < len(idx):
            idx = random.sample(idx, num_negative)
        negative_samples[i, torch.LongTensor(idx)] = True
    return negative_samples


def scale_audio_length(start, end, config):
    for kernel, stride in zip(config.conv_kernel, config.conv_stride):
        start = (start - kernel) // stride + 1
        end = (end - kernel) // stride + 1
    return [start, end]


def group_scale_audio_length(arr, config):
    for kernel, stride in zip(config.conv_kernel, config.conv_stride):
        arr = torch.div(arr - kernel, stride, rounding_mode="floor") + 1
    return arr


def similarity(x, y, t, mode="cross_attn"):
    if mode == "cosine":
        return torch.mm(normalize(x), normalize(y).transpose(0, 1)) / t
    if mode == "distance":
        return -torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1) / t
    if mode == "diag_cosine":
        return (normalize(x) * normalize(y)).sum(-1) / t
    if mode == "diag_cross_attn":
        return (x * y).sum(-1) / t
    return torch.mm(x, y.transpose(0, 1)) / t


def compute_valid_for_rm(sequences, length, pooling_mode):
    if pooling_mode == "first":
        valid = [0 for _ in range(length)]
    else:
        valid = [[0 for _ in range(length)] for _ in sequences]
    for i, item in enumerate(sequences):
        start, end = item
        if pooling_mode == "first":
            valid[start] = 1
        else:
            for j in range(start, end):
                valid[i][j] = 1
    return torch.BoolTensor(valid)
