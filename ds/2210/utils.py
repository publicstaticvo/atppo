import os
import torch.distributed as dist
from transformers import PretrainedConfig, WavLMConfig, RobertaConfig

CUSTOM_CONFIG_NAME = "config.json"
AUDIO_CONFIG_NAME = "audio_config.json"
TEXT_CONFIG_NAME = "text_config.json"


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


class ATConfig(PretrainedConfig):
    audio_config_cls = WavLMConfig
    text_config_cls = RobertaConfig

    def __init__(self):
        super().__init__()
        self.text = self.audio_config_cls()
        self.audio = self.text_config_cls()

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.audio.to_json_file(os.path.join(save_directory, AUDIO_CONFIG_NAME), True)
        self.text.to_json_file(os.path.join(save_directory, TEXT_CONFIG_NAME), True)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, return_kwargs=True, **kwargs):
        config = cls.from_json_files(os.path.join(pretrained_model_name_or_path, AUDIO_CONFIG_NAME),
                                     os.path.join(pretrained_model_name_or_path, TEXT_CONFIG_NAME))
        if return_kwargs:
            return config, kwargs
        return config

    @classmethod
    def from_configs(cls, audio, text):
        config = cls()
        config.audio = audio
        config.text = text
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
