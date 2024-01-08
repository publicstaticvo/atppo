import torch
from models import ATModel
from rm_trainer import AlignTrainer
from util import ATConfig, similarity


class WordAlignTrainer(AlignTrainer):

    def __init__(self, config: ATConfig, audio=None, text=None):
        config.fused.num_hidden_layers = 0
        super(WordAlignTrainer, self).__init__(config, ATModel, audio, text)

    def reward_loss(self, audio, text, audio_valid, text_valid, negative_indices=None):
        bs = audio.shape[0]
        losses = 0
        for i in range(bs):
            # a_valid和t_valid的大小为N*H
            a_valid = self.valid_filter(audio[i], audio_valid[i], self.config.audio.pooling_mode)
            t_valid = self.valid_filter(text[i], text_valid[i], self.config.text.pooling_mode)
            sim = similarity(t_valid, a_valid, self.temperature)  # N*N
            positive = torch.diag(sim)
            negative = torch.sum(sim * negative_indices[i], dim=-1) / torch.clamp_min(torch.sum(negative_indices[i], dim=-1), 1)
            loss = -torch.log(torch.sigmoid(positive - negative)).mean()
            # loss = torch.mean(-torch.log(positive / (positive + negative)))
            losses += loss
        return losses / bs

    def forward(self, audio_input, text_input, audio_mask, text_mask, turn_id=None, audio_valid=None, text_valid=None, neg=None, mlm_label=None):
        features, mam_label, a_masked = self.model(audio_input, text_input, audio_mask, text_mask, turn_id, self.perform_mlm)
        audio_features, _, text_features = features
        rm_loss = self.reward_loss(audio_features, text_features, audio_valid, text_valid, neg)
        if self.perform_mlm:
            mlm = self.mlm_loss(text_features, mlm_label)
            mam = self.mam_loss(audio_features, mam_label, a_masked)
            return mlm, mam, rm_loss
        return 0, 0, rm_loss
