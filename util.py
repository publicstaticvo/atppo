import torch
import random
import torch.distributed as dist


def pad(sequence, length):
    seq_len = sequence.shape[0]
    att = torch.cat([torch.ones_like(sequence), torch.zeros(length - seq_len, dtype=sequence.dtype)])
    sequence = torch.cat([sequence, torch.zeros(length - seq_len, dtype=sequence.dtype)])
    return sequence, att.to(dtype=torch.long)


def pad_cut(sequence, length, pad_token=0):
    seq_len = sequence.shape[0]
    if length > seq_len:
        padding = torch.ones(length - seq_len, dtype=sequence.dtype) * pad_token
        sequence = torch.cat([sequence, padding])
        att = torch.cat([torch.ones(seq_len, dtype=torch.long), padding.long()])
    else:
        if sequence.dtype == torch.long:
            sequence = torch.cat([sequence[:1], sequence[1 - length:]])
        else:
            sequence = sequence[:length]
        att = torch.ones(length, dtype=torch.long)
    return sequence, att


def scale_audio_length(start, end, config):
    for kernel, stride in zip(config.conv_kernel, config.conv_stride):
        start = (start - kernel) // stride + 1
        end = (end - kernel) // stride + 1
    return [start, end]


def group_scale_audio_length(arr, config):
    for kernel, stride in zip(config.conv_kernel, config.conv_stride):
        arr = torch.div(arr - kernel, stride, rounding_mode="floor") + 1
    return arr


def compute_valid_for_tpp(transcript, offset, length, mode, audio_length):
    sv = [0 for _ in range(length)]
    ev = [0 for _ in range(length)]
    start_labels, end_labels = [], []
    for i, item in enumerate(transcript):
        sv[offset + item[-4]] = 1
        ev[offset + item[-3] - 1] = 1
        sl, el = float(f"{item[-2] / audio_length:.3f}"), float(f"{item[-1] / audio_length:.3f}")
        if mode:
            start_labels.append(int(sl * 100))
            end_labels.append(int(el * 100) - 1)
        else:
            start_labels.append(sl)
            end_labels.append(el)
    return torch.BoolTensor(sv), torch.BoolTensor(ev), start_labels, end_labels


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
    return torch.BoolTensor(valid), len(sequences)


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def negative_sampling(words, num_negative):
    words = [x[0][:-1] if x[0][-1] in [',', '.', '?', '!'] and len(x[0]) > 1 else x[0] for x in words]
    negative_samples = torch.zeros([len(words), len(words)], dtype=torch.bool)
    for i, x in enumerate(words):
        idx = [j for j, y in enumerate(words) if y != x]
        if 0 < num_negative < len(idx):
            idx = random.sample(idx, num_negative)
        negative_samples[i, torch.LongTensor(idx)] = True
    return negative_samples


def get_train_ds_config(train_batch_size, num_gpus, grad_acc=1, stage=2, fp16_level=2, offload=False):
    return {
        "train_batch_size": train_batch_size * num_gpus * grad_acc,
        "train_micro_batch_size_per_gpu": train_batch_size,
        "zero_optimization": {
            "stage": stage,
            "offload_param": {
                "device": "cpu" if offload else "none"
            },
            "offload_optimizer": {
                "device": "cpu" if offload else "none"
            },
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 3e7,
            "stage3_prefetch_bucket_size": 3e7,
            "memory_efficient_linear": False
        },
        "fp16": {
            "enabled": True,
            "opt_level": f"O{fp16_level}",
            "loss_scale_window": 200
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }


def get_eval_ds_config(train_batch_size, stage=2, fp16_level=2, offload=False):
    return {
        "train_micro_batch_size_per_gpu": train_batch_size,
        "zero_optimization": {
            "stage": stage,
            "stage3_param_persistence_threshold": 1e4,
            "offload_param": {
                "device": "cpu" if offload else "none"
            }
        },
        "fp16": {
            "enabled": True,
            "opt_level": f"O{fp16_level}"
        },
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }
