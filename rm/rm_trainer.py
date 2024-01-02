import math
import torch
from torch import nn
from model import ATModel
from configuration_at import ATConfig
from transformers import PreTrainedModel
from torch.nn.functional import normalize
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaEncoder
from wavlm import WavLMMAMHead, WavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder


def similar(x, y, t, mode="cosine"):
    if mode == "cosine":
        return torch.mm(normalize(x), normalize(y).transpose(0, 1)) / t
    if mode == "distance":
        return -torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1) / t
    return torch.mm(x, y.transpose(0, 1)) / t


class ATRewardModel(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_unexpected = [r"fused_encoder"]
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (RobertaEncoder, WavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder)):
            module.gradient_checkpointing = value

    def __init__(self, config: ATConfig, audio=None, text=None):
        super(ATRewardModel, self).__init__(config)
        self.hidden_size = config.text.hidden_size
        self.model = ATModel(config, audio, text, tpp=False)
        self.num_negative = config.num_negative
        self.perform_mlm = config.perform_mlm
        if self.perform_mlm:
            self.mlm_head = RobertaLMHead(config.text)
            self.mam_head = WavLMMAMHead(self.hidden_size, config.audio.conv_dim[-1])
            self.vocab_size = config.text.vocab_size
            self.ce = torch.nn.CrossEntropyLoss()
            self.l1 = torch.nn.L1Loss()
        # self.start_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))
        # self.end_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))
        self.temperature = 1

    def mlm_loss(self, text, label):
        mlm_pre = self.mlm_head(text)
        mlm_loss = self.ce(mlm_pre.view(-1, self.vocab_size), label.view(-1))  # 未mask的位置，label为-100。
        if torch.isnan(mlm_loss):
            mlm_loss = torch.tensor(0.0, device=text.device)
        return mlm_loss

    def mam_loss(self, audio, label, masked_indices):
        mam_pre = self.mam_head(audio)
        mam_loss = torch.tensor(0.0, device=audio.device)
        if torch.sum(masked_indices[1]) != 0:
            mam_loss = self.l1(mam_pre.masked_select(masked_indices[0]), label.masked_select(masked_indices[1]))
        return mam_loss

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
            sim = similar(t_valid, a_valid, self.temperature)  # N*N
            positive = torch.diag(sim)
            negative = torch.sum(sim * negative_indices[i], dim=-1) / torch.clamp_min(torch.sum(negative_indices[i], dim=-1), 1)
            loss = -torch.log(torch.sigmoid(positive - negative)).mean()
            # loss = torch.mean(-torch.log(positive / (positive + negative)))
            losses += loss
        return losses / bs

    def forward_features_for_ppo(self, audio_input, text_input, audio_mask, text_mask, turn_id=None, text_valid=None):
        audio_features, text_features, _, _ = self.model(audio_input, text_input, audio_mask, text_mask, turn_id, False)
        text_words = [self.valid_filter(text_features[i], text_valid[i], self.config.text.pooling_mode) for i in range(text_features.shape[0])]
        return audio_features, text_words

    def forward(self, audio_input, text_input, audio_mask, text_mask, turn_id=None, audio_valid=None, text_valid=None, neg=None, mlm_label=None):
        audio_features, text_features, mam_label, a_masked = self.model(audio_input, text_input, audio_mask, text_mask, turn_id, self.perform_mlm)
        rm_loss = self.word_level_contrastive(audio_features, text_features, audio_valid, text_valid, neg)
        if self.perform_mlm:
            mlm = self.mlm_loss(text_features, mlm_label)
            mam = self.mam_loss(audio_features, mam_label, a_masked)
            return mlm, mam, rm_loss
        return 0, 0, rm_loss
