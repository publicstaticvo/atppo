import os
import sys
import tqdm
import argparse
import numpy as np
from apex import amp
from torch.nn.functional import normalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from util import *
from models import ATMultiTurnModel, WavLMForMultiTurn
from dataset import ATDataset, DataCollatorForDP
from transformers import RobertaTokenizerFast
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
SAMPLE_RATE = 16000


def construct_mean_map(d):
    dd = torch.arange(d)
    dd = dd[None, :] + 1 - dd[:, None]
    return torch.clamp_min(dd, 1).unsqueeze(-1)


def split_audio_features(audio, audio_attention_mask):
    bs = audio_attention_mask.shape[0] // 2
    audio_attention_mask_sum = torch.clamp_max(audio_attention_mask.view(bs, 2, -1).sum(-1).div(320, rounding_mode='trunc'), 99)
    audio_attention_mask_sum = audio_attention_mask_sum.long().tolist()
    af = []
    for i in range(bs):
        af.append(audio[i, 1:audio_attention_mask_sum[i][0]+1])
        af.append(audio[i, audio_attention_mask_sum[i][0]+2:])
        assert af[-1].shape[0] == audio_attention_mask_sum[i][1]
    return af


def split_text_words(text, split_mark):
    tw = []
    for tt, sm in zip(text, split_mark):
        tw.append(tt[:sm])
        tw.append(tt[sm:])
    return tw


def precompute_max_sim(s):
    # Precompute the maximum of s(l, i, j) for all i, j and every possible k ≤ l ≤ i
    m, n = s.shape[-2:]
    max_s = torch.zeros_like(s).tolist()
    argmax_s = torch.zeros_like(s).tolist()
    s = s.tolist()
    for j in range(n):
        for i in range(m):
            max_s[0][i][j] = s[0][i][j]
            argmax_s[0][i][j] = 0
            for k in range(1, i + 1):
                if max_s[k-1][i][j] < s[k][i][j]:
                    max_s[k][i][j] = s[k][i][j]
                    argmax_s[k][i][j] = k
                else:
                    max_s[k][i][j] = max_s[k-1][i][j]
                    argmax_s[k][i][j] = argmax_s[k-1][i][j]
    return s, max_s, argmax_s


if __name__ == "__main__":
    # 1.输入参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--apex_level", default=2, type=int)
    parser.add_argument("--audio_length", default=10, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--file_prefix", default=None, type=str)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--num_ends", default=1, type=int)
    parser.add_argument("--num_turns", default=8, type=int)
    parser.add_argument("--reward_path", default=None, type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--text_length", default=512, type=int)
    parser.add_argument("--tokenizer_path", default=None, type=str, required=True)
    parser.add_argument("--transcripts", default=None, type=str, required=True)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", init_method='env://')
    # 2.设随机数
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # 3。读输入数据
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_path)
    train_data = ATDataset(args.transcripts, args.num_turns, args.file_prefix)
    # 4。建立模型
    model = ATMultiTurnModel.from_pretrained(args.reward_path).to(args.device)
    model = amp.initialize(model, opt_level=f"O{args.apex_level}", keep_batchnorm_fp32=False if args.apex_level >= 2 else None, loss_scale="dynamic")
    c = DataCollatorForDP(args, tokenizer)
    if args.local_rank >= 0:
        train_loader = DataLoader(train_data, sampler=DistributedSampler(train_data, seed=args.seed), batch_size=args.batch_size, collate_fn=c, pin_memory=True, shuffle=False)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=c, sampler=SequentialSampler(train_data), shuffle=False)
    mean_map = construct_mean_map(200).to(args.device).half()
    for j, batch in enumerate(tqdm.tqdm(train_loader)):
        batch = to_device(batch, args.device)
        audio_features, text_words = model(**batch)
        audio_features = split_audio_features(audio_features, batch["audio_attention_mask"])
        text_words = split_text_words(text_words, batch["split_marks"])  # audio和text均为2B个
        tpp_starts, tpp_ends = [], []  # 一个batch内部所有label打成一个1D数组
        for a, t in zip(audio_features, text_words):
            # a: M*H t: N*H
            m = a.shape[0]
            n = t.shape[0]
            c = a.unsqueeze(1).repeat(1, m, 1).permute(2, 0, 1).triu().permute(1, 2, 0) / mean_map[:m, :m, :]
            sim = torch.einsum("ijk,lk->ijl", normalize(c), normalize(t))  # M*M*N
            # print(sim.tolist())
            sim, max_sim, argmax_sim = precompute_max_sim(sim)
            # 动态规划
            obj = [[float("-inf") for _ in range(n)] for _ in range(m)]
            starts = [[0 for _ in range(n)] for _ in range(m)]  # 每次被选中的argmax_sim，由end_labels[i]指向start_labels[i]
            ends = [[0 for _ in range(n)] for _ in range(m)]  # 由end_labels[i]指向end_labels[i-1]
            for j in range(n):  # 前j个词
                for i in range(m):  # 前i个语音token
                    if j == 0:
                        obj[i][j] = sim[0][i][j]
                        starts[i][j] = 0
                        ends[i][j] = -1
                    else:
                        for k in range(i):
                            if obj[k][j - 1] + max_sim[k][i][j] > obj[i][j]:
                                obj[i][j] = obj[k][j - 1] + max_sim[k][i][j]
                                starts[i][j] = argmax_sim[k][i][j]
                                ends[i][j] = k
            # 最大匹配值
            mm, argmax = float("-inf"), -1
            for i, o in enumerate(obj):
                if mm < o[-1]:
                    mm = o[-1]
                    argmax = i
            # 回溯路径
            s, e = [], [argmax]
            for j in reversed(range(n)):
                if j > 0:
                    e.append(ends[e[-1]][j])
                s.append(starts[e[-1]][j])
            tpp_starts.extend(s[::-1])
            tpp_ends.extend(e[::-1])
        print(tpp_starts, tpp_ends)
