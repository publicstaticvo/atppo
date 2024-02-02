import torch
from torch import nn
from util import ATConfig
from models import ATForTPP, ATMultiTurnModel, TrainerBase


class ReconstructTrainer(TrainerBase):

    def __init__(self, config: ATConfig, audio=None, text=None, *args, **kwargs):
        super(ReconstructTrainer, self).__init__(config)
        self.num_ends = config.fused.num_ends
        self.train_phase = config.train_phase
        model_class = ATMultiTurnModel if config.train_phase == 1 else ATForTPP
        self.model = model_class(config, audio=audio, text=text)
        self.start_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))
        self.end_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))

    def forward(self, audio_input, text_input, audio_attention_mask, text_attention_mask,
                head_mask_for_fused=None, turn_id=None, output_attentions=False):
        # audio: 3B * 160000  text: 2B * 514  mlm_labels: B * 514  turn_id: B * 514
        outputs = self.model(audio_input, text_input, audio_attention_mask, text_attention_mask, turn_id,
                             output_attentions=output_attentions, head_mask_for_fused=head_mask_for_fused)
        bs, text_len = text_input.shape
        if output_attentions:
            return outputs[0][1], text_len
        else:
            fused_features, mam_label, (_, attention_mask) = outputs
        text_features = fused_features[:, :text_len]
        audio_features = fused_features[:, text_len:]
        mlm = self.mlm_loss(text_features, text_input)
        # reconstruct mam_label into 2-turn
        # print(audio_features.shape, attention_mask.shape, mam_label.shape)
        audio_features = audio_features.reshape(bs * 2, 100, self.hidden_size)[:, 1:]
        mam_pre = self.mam_head(audio_features)
        attention_mask = attention_mask.bool().unsqueeze(-1)
        mam_pre = mam_pre.masked_select(attention_mask)
        mam_label = mam_label.masked_select(attention_mask)
        mam = self.l1(mam_pre, mam_label.detach())
        return mlm, mam, 0
