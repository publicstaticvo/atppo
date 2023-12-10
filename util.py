import os
import copy
import torch
import pickle
import torch.distributed as dist
from transformers import PretrainedConfig, WavLMConfig, RobertaConfig

CUSTOM_CONFIG_NAME = "config.json"
AUDIO_CONFIG_NAME = "audio_config.json"
TEXT_CONFIG_NAME = "text_config.json"


def pad(sequence, length):
    seq_len = sequence.shape[0]
    att = torch.cat([torch.ones_like(sequence), torch.zeros(length - seq_len, dtype=sequence.dtype)])
    sequence = torch.cat([sequence, torch.zeros(length - seq_len, dtype=sequence.dtype)])
    return sequence, att.to(dtype=torch.long)


def pad_cut(sequence, length, pad_token=0):
    seq_len = sequence.shape[0]
    if length > seq_len:
        padding = torch.ones(length - seq_len, dtype=sequence.dtype) * pad_token
        att = torch.cat([torch.ones_like(sequence), padding])
        sequence = torch.cat([sequence, padding])
    else:
        if sequence.dtype == torch.long:
            sequence = torch.cat([sequence[:1], sequence[1 - length:]])
        else:
            sequence = sequence[:length]
        att = torch.ones_like(sequence)
    return sequence, att


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def read_processed_pretrain(combined_path):
    if os.path.isdir(combined_path):
        datas = None
        for r, d, fs in os.walk(combined_path):
            if not d:
                for f in fs:
                    with open(os.path.join(r, f), "rb") as fp:
                        if datas is None:
                            datas = pickle.load(fp)
                        else:
                            datas += pickle.load(fp)
    else:
        with open(combined_path, "rb") as f:
            datas = pickle.load(f)
    return datas


class ATConfig(PretrainedConfig):
    audio_config_cls = WavLMConfig
    text_config_cls = RobertaConfig

    def __init__(self):
        super().__init__()
        self.text = self.text_config_cls()
        self.audio = self.audio_config_cls()
        self.fused = self.text_config_cls()

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.audio.to_json_file(os.path.join(save_directory, AUDIO_CONFIG_NAME), True)
        self.text.to_json_file(os.path.join(save_directory, TEXT_CONFIG_NAME), True)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs):
        config = cls.from_json_files(os.path.join(pretrained_model_name_or_path, AUDIO_CONFIG_NAME),
                                     os.path.join(pretrained_model_name_or_path, TEXT_CONFIG_NAME))
        if not return_unused_kwargs or len(kwargs) == 0:
            return config
        return config, kwargs

    @classmethod
    def from_configs(cls, audio, text):
        config = cls()
        config.audio = audio
        config.text = text
        config.fused = copy.deepcopy(text)
        return config

    @classmethod
    def from_json_files(cls, audio, text):
        return cls.from_configs(cls.audio_config_cls.from_json_file(audio), cls.text_config_cls.from_json_file(text))

    def set_pooling_mode(self, audio, text):
        self.text.pooling_mode = text
        self.audio.pooling_mode = audio

    def set_length(self, audio, text):
        self.text.max_length = text
        self.audio.max_length = audio


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
