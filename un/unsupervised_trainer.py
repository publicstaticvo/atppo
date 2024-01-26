import torch
from torch import nn
from util import ATConfig
from models import ATForTPP, ATModel, TrainerBase


class UnsupervisedTrainer(TrainerBase):

    def __init__(self, config: ATConfig, audio=None, text=None, *args, **kwargs):
        super(UnsupervisedTrainer, self).__init__(config)
        self.num_ends = config.fused.num_ends
        self.train_phase = config.train_phase
        model_class = ATModel if config.train_phase == 1 else ATForTPP
        self.model = model_class(config, audio=audio, text=text)
        self.start_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))
        self.end_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))

    def forward(self, audio_input, text_input, audio_attention_mask, text_attention_mask, mlm_labels=None,
                turn_id=None, start_valid=None, end_valid=None):
        # audio: 3B * 160000  text: 2B * 514  mlm_labels: B * 514  turn_id: B * 514
        fused_features, mam_label, a_masked = self.model(audio_input, text_input, audio_attention_mask, text_attention_mask, turn_id, mlm_labels is not None)
        bs, text_len = mlm_labels.shape
        rs_loss = 0
        if self.train_phase == 2:
            fused_features = fused_features.view(bs, 4, -1, self.hidden_size)
            text_features = fused_features[:, 0, :text_len]
            audio_features = fused_features[:, 0, text_len:]
            rs_loss = self.response_selection(fused_features, bs)
        else:
            text_features = fused_features[:, :text_len]
            audio_features = fused_features[:, text_len:]
        mlm = self.mlm_loss(text_features, mlm_labels)
        mam = self.mam_loss(audio_features, mam_label, a_masked)
        return mlm, mam, rs_loss
