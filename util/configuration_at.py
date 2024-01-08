import os
import copy
from transformers import PretrainedConfig, WavLMConfig, RobertaConfig

CUSTOM_CONFIG_NAME = "config.json"
AUDIO_CONFIG_NAME = "audio_config.json"
TEXT_CONFIG_NAME = "text_config.json"
FUSED_CONFIG_NAME = "fused_layer.json"


class ATConfig(PretrainedConfig):
    audio_config_cls = WavLMConfig
    text_config_cls = RobertaConfig

    def __init__(self):
        super().__init__()
        self.text = self.text_config_cls()
        self.audio = self.audio_config_cls()
        self.fused = self.text_config_cls()

    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.audio.to_json_file(os.path.join(save_directory, AUDIO_CONFIG_NAME), True)
        self.text.to_json_file(os.path.join(save_directory, TEXT_CONFIG_NAME), True)
        self.fused.to_json_file(os.path.join(save_directory, FUSED_CONFIG_NAME), True)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs):
        config = cls.from_json_files(os.path.join(pretrained_model_name_or_path, AUDIO_CONFIG_NAME),
                                     os.path.join(pretrained_model_name_or_path, TEXT_CONFIG_NAME),
                                     os.path.join(pretrained_model_name_or_path, FUSED_CONFIG_NAME))
        if not return_unused_kwargs or len(kwargs) == 0:
            return config
        return config, kwargs

    @classmethod
    def from_configs(cls, audio, text, fused=None):
        config = cls()
        config.audio = audio
        config.text = text
        config.fused = copy.deepcopy(text) if fused is None else fused
        return config

    @classmethod
    def from_json_files(cls, audio, text, fused=None):
        audio = cls.audio_config_cls.from_json_file(audio)
        text = cls.text_config_cls.from_json_file(text)
        fused = cls.text_config_cls.from_json_file(fused) if fused is not None and os.path.isfile(fused) else None
        return cls.from_configs(audio, text, fused)

    def set_pooling_mode(self, audio, text):
        self.text.pooling_mode = text
        self.audio.pooling_mode = audio

    def set_length(self, audio, text):
        self.text.max_length = text
        self.audio.max_length = audio
