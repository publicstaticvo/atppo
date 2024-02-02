import torch
from torch import nn
from util import ATConfig
from models import ATForTPP, ATMultiTurnModel, TrainerBase, AT23062682


class MaskedLMTrainer(TrainerBase):

    def __init__(self, config: ATConfig, audio=None, text=None, *args, **kwargs):
        super(MaskedLMTrainer, self).__init__(config)
        self.num_ends = config.fused.num_ends
        self.train_phase = config.train_phase
        model_class = AT23062682 if config.train_phase == 1 else ATForTPP
        self.model = model_class(config, audio=audio, text=text)
        self.start_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))
        self.end_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))

    def forward(self, audio_input, text_input, audio_attention_mask, text_attention_mask, mlm_labels=None,
                head_mask_for_fused=None, perform_mam=None, turn_id=None, output_attentions=False):
        # audio: 3B * 160000  text: 2B * 514  mlm_labels: B * 514  turn_id: B * 514
        if perform_mam is None:
            perform_mam = mlm_labels is not None
        outputs = self.model(audio_input, text_input, audio_attention_mask, text_attention_mask, turn_id,
                             masked_modeling=perform_mam, output_attentions=output_attentions,
                             head_mask_for_fused=head_mask_for_fused)
        bs, text_len = mlm_labels.shape
        if output_attentions:
            # (fused_features, attentions), mam_label, a_masked = outputs
            return outputs[0][1], text_len
        else:
            fused_features, mam_label, a_masked = outputs
        mam, rs_loss = 0, 0
        if self.train_phase == 2:
            fused_features = fused_features.view(bs, 4, -1, self.hidden_size)
            text_features = fused_features[:, 0, :text_len]
            audio_features = fused_features[:, 0, text_len:]
            rs_loss = self.response_selection(fused_features, bs)
        else:
            text_features = fused_features[:, :text_len]
            audio_features = fused_features[:, text_len:]
        mlm = self.mlm_loss(text_features, mlm_labels)
        if perform_mam:
            mam = self.mam_loss(audio_features, mam_label, a_masked)
        return mlm, mam, rs_loss
