import torch
from torch import nn
from model import ATModel
from configuration_at import ATConfig
from transformers import PreTrainedModel
from wavlm import WavLMMAMHead, WavLMForMultiTurn, WavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaEncoder, RobertaModel, RobertaLayer


class ATForTPP(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_missing = ["mlm_head", "mam_head", "selection_head", "start_prediction_head", "end_prediction_head"]
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (RobertaEncoder, WavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder)):
            module.gradient_checkpointing = value

    def __init__(self, config: ATConfig, audio=None, text=None):
        super(ATForTPP, self).__init__(config)
        self.hidden_size = config.text.hidden_size
        self.num_ends = config.fused.num_ends
        self.model = ATModel(config, audio, text)
        self.mlm_head = RobertaLMHead(config.text)
        self.mam_head = WavLMMAMHead(self.hidden_size, config.audio.conv_dim[-1])
        self.selection_head = nn.Linear(self.hidden_size, 4)
        self.start_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))
        self.end_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))
        self.vocab_size = config.text.vocab_size
        # self.conv_dim = config.audio.conv_dim[-1]
        self.ce = torch.nn.CrossEntropyLoss()
        self.l1 = torch.nn.L1Loss()
        self.temperature = 1 / 15

    def tpp_loss(self, text_fused, start_valid=None, end_valid=None, starts=None, ends=None):
        words = text_fused.masked_select(start_valid.unsqueeze(-1)).view(-1, self.hidden_size)
        pred_start = self.start_prediction_head(words).squeeze(-1)
        words = text_fused.masked_select(end_valid.unsqueeze(-1)).view(-1, self.hidden_size)
        pred_end = self.end_prediction_head(words).squeeze(-1)
        if self.num_ends == 1:
            return torch.mean(torch.pow(torch.cat([starts, ends]) - torch.cat([pred_start, pred_end]), 2))
        return self.ce(torch.cat([pred_start, pred_end], dim=0), torch.cat([starts, ends]))

    def forward(self, audio_input, text_input, audio_attention_mask, text_attention_mask, mlm_label=None,
                turn_id=None, start_valid=None, end_valid=None, starts=None, ends=None):
        # audio: 3B * 160000  text: 2B * 514  mlm_label: B * 514  turn_id: B * 514
        fused_features, mam_label, a_masked = self.model(audio_input, text_input, audio_attention_mask, text_attention_mask, turn_id, mlm_label is not None)
        bs, text_len = mlm_label.shape
        fused_input = fused_features.view(bs, 4, -1, self.hidden_size)
        mam_pre = self.mam_head(fused_input[:, 0, text_len:])
        text_fused = fused_input[:, 0, :text_len]
        mlm_pre = self.mlm_head(text_fused)
        mlm_loss = self.ce(mlm_pre.view(-1, self.vocab_size), mlm_label.view(-1))  # 未mask的位置，label为-100。
        if torch.isnan(mlm_loss):
            mlm_loss = torch.tensor(0.0, device=text_input.device)
        mam_loss = torch.tensor(0.0, device=text_input.device)
        if torch.sum(a_masked[1]) != 0:
            mam_loss = self.l1(mam_pre.masked_select(a_masked[0]), mam_label.masked_select(a_masked[1]))
        response_select = self.selection_head(fused_input[:, :, 0].view(4 * bs, self.hidden_size))
        rs_loss = self.ce(response_select, torch.arange(4).to(mlm_pre.device).repeat(bs))
        span_loss = self.tpp_loss(text_fused, start_valid, end_valid, starts, ends)
        return mlm_loss, mam_loss, rs_loss, span_loss
