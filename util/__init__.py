import torch
import random
import torch.distributed as dist

from .parallel import DDP
from .configuration_at import ATConfig
from .tpp_util import compute_valid_for_tpp
from .ppo_util import step, concat_audio
from .word_rm import *
from .sent_rm import *


def pad(sequence, length):
    seq_len = sequence.shape[0]
    att = torch.cat([torch.ones_like(sequence), torch.zeros(length - seq_len, dtype=sequence.dtype)])
    sequence = torch.cat([sequence, torch.zeros(length - seq_len, dtype=sequence.dtype)])
    return sequence, att.to(dtype=torch.long)


def pad_cut(sequence, length, pad_token=0):
    seq_len = sequence.shape[0]
    device = sequence.device
    if length > seq_len:
        padding = torch.ones(length - seq_len, dtype=sequence.dtype, device=device) * pad_token
        sequence = torch.cat([sequence, padding])
        att = torch.cat([torch.ones(seq_len, dtype=torch.long, device=device), padding.long()])
    else:
        if sequence.dtype == torch.long:
            sequence = torch.cat([sequence[:1], sequence[1 - length:]])
        else:
            sequence = sequence[:length]
        att = torch.ones(length, dtype=torch.long, device=device)
    return sequence, att


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


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


__all__ = ['DDP',
           'dist',
           'step',
           'torch',
           'random',
           'pad_cut',
           'ATConfig',
           'get_rank',
           'similarity',
           'concat_audio',
           'negative_audio',
           'negative_sampling',
           'scale_audio_length',
           'get_eval_ds_config',
           'get_train_ds_config',
           'compute_valid_for_rm',
           'compute_valid_for_tpp',
           'construct_audio_batch',
           'group_scale_audio_length'
           ]
