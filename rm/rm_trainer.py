import math
import torch
from torch import nn
from model import ATModel
from util import ATConfig
from transformers import PreTrainedModel
from torch.nn.functional import normalize
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaEncoder
from wavlm import WavLMMAMHead, WavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder


def similar(x, y, t=1.0, mode="cosine"):
    if mode == "cosine":
        return torch.mm(normalize(x), normalize(y).transpose(0, 1)) / t
    if mode == "distance":
        return -torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1) / t
    return torch.mm(x, y.transpose(0, 1)) / t


class ATRewardModel(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_missing = ["mlm_head", "mam_head", "selection_head", "start_prediction_head", "end_prediction_head"]
    # _keys_to_ignore_on_save = ["mlm_head", "mam_head", "selection_head", "start_prediction_head", "end_prediction_head"]
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (RobertaEncoder, WavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder)):
            module.gradient_checkpointing = value

    def __init__(self, config: ATConfig, audio=None, text=None):
        super(ATRewardModel, self).__init__(config)
        self.hidden_size = config.text.hidden_size
        self.model = ATModel(config, audio, text)
        self.mlm_head = RobertaLMHead(config.text)
        self.mam_head = WavLMMAMHead(self.hidden_size, config.audio.conv_dim[-1])
        self.selection_head = nn.Linear(self.hidden_size, 4)
        self.start_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, 1))
        self.end_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, 1))
        self.vocab_size = config.text.vocab_size
        self.num_negative = config.num_negative
        # self.conv_dim = config.audio.conv_dim[-1]
        self.ce = torch.nn.CrossEntropyLoss()
        self.l1 = torch.nn.L1Loss()
        self.temperature = 1 / 15

    def valid_filter(self, outputs, valid, pooling_mode):
        words = valid.shape[0]
        if pooling_mode == "first":
            # 每个valid形状为L
            return outputs.masked_select(valid.unsqueeze(-1)).view(-1, self.hidden_size)
        elif pooling_mode == "mean":
            # 每个valid形状为Ni*L
            temp = outputs.unsqueeze(0).repeat(words, 1, 1).masked_fill(~valid.unsqueeze(-1), 0)
            return torch.sum(temp, dim=1) / torch.sum(valid, dim=1, keepdim=True)

    def word_level_contrastive(self, audio, text, audio_valid, text_valid, negative_indices=None):
        bs = audio.shape[0]
        losses = 0
        for i in range(bs):
            # a_valid和t_valid的大小为N*H
            a_valid = self.valid_filter(audio[i], audio_valid[i], self.config.audio.pooling_mode)
            t_valid = self.valid_filter(text[i], text_valid[i], self.config.text.pooling_mode)
            sim = torch.exp(similar(t_valid, a_valid, self.temperature))  # N*N
            positive = torch.diag(sim)
            negative = torch.sum(sim * negative_indices[i].unsqueeze(-1), dim=-1)
            losses += -torch.log(positive / (positive + negative))
        return losses / bs

    def forward(self, audio_input, text_input, audio_mask, text_mask, turn_id=None, audio_valid=None, text_valid=None, neg=None):
        fused_features = self.model(audio_input, text_input, audio_mask, text_mask, turn_id)
        bs, text_len = text_input.shape
        text_features = fused_features[:, :text_len]
        audio_features = fused_features[:, text_len:]
        loss = self.word_level_contrastive(audio_features, text_features, audio_valid, text_valid, neg)
        return loss
