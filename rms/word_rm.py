import torch
from rm_trainer import AlignTrainer
from util import ATConfig, similarity
from models import ATSingleTurnModel


class WordAlignTrainer(AlignTrainer):

    def __init__(self, config: ATConfig, audio=None, text=None):
        # config.fused.num_hidden_layers = 0
        super(WordAlignTrainer, self).__init__(config, ATSingleTurnModel, audio, text)

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

    def forward(self, audio_input, text_input, audio_mask, text_mask, turn_id=None, audio_valid=None, text_valid=None, neg=None):
        audio_features, _, text_features = self.model(audio_input, text_input, audio_mask, text_mask, turn_id)
        rm_loss = self.reward_loss(audio_features, text_features, audio_valid, text_valid, neg)
        return rm_loss
