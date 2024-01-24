import torch
from torch import nn
from util import ATConfig
from models import ATForTPP, TrainerBase


class TPPTrainer(TrainerBase):
    _keys_to_ignore_on_load_missing = ["mlm_head", "mam_head", "selection_head", "start_prediction_head", "end_prediction_head"]

    def __init__(self, config: ATConfig, audio=None, text=None, *args, **kwargs):
        super(TPPTrainer, self).__init__(config)
        self.num_ends = config.fused.num_ends
        self.model = ATForTPP(config, audio=audio, text=text)
        self.start_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))
        self.end_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))

    def tpp_loss(self, text_fused, start_valid=None, end_valid=None, starts=None, ends=None):
        words = text_fused.masked_select(start_valid.unsqueeze(-1)).view(-1, self.hidden_size)
        pred_start = self.start_prediction_head(words).squeeze(-1)
        words = text_fused.masked_select(end_valid.unsqueeze(-1)).view(-1, self.hidden_size)
        pred_end = self.end_prediction_head(words).squeeze(-1)
        pred = torch.cat([pred_start, pred_end], dim=0)
        if starts is None:
            return pred
        if self.num_ends == 1:
            return torch.mean(torch.pow(pred - torch.cat([starts, ends]), 2)), pred
        return self.ce(pred, torch.cat([starts, ends])), pred

    def forward(self, audio_input, text_input, audio_attention_mask, text_attention_mask, mlm_label=None,
                turn_id=None, start_valid=None, end_valid=None, starts=None, ends=None):
        # audio: 3B * 160000  text: 2B * 514  mlm_label: B * 514  turn_id: B * 514
        fused_features, mam_label, a_masked = self.model(audio_input, text_input, audio_attention_mask, text_attention_mask, turn_id, mlm_label is not None)
        bs, text_len = mlm_label.shape
        fused_input = fused_features.view(bs, 4, -1, self.hidden_size)
        text_fused = fused_input[:, 0, :text_len]
        mlm = self.mlm_loss(text_fused, mlm_label)
        mam = self.mam_loss(fused_input[:, 0, text_len:], mam_label, a_masked)
        rs_loss = self.response_selection(fused_input, bs)
        span_loss, _ = self.tpp_loss(text_fused, start_valid, end_valid, starts, ends)
        return mlm, mam, rs_loss, span_loss
