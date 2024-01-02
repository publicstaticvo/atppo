import torch
from torch import nn
from model import ATModel
from util import ATConfig
from transformers import PreTrainedModel
from wavlm import WavLMMAMHead, WavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaEncoder


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
        self.step_count = 0

    def tpp_loss(self, text_fused, start_valid=None, end_valid=None, starts=None, ends=None):
        self.step_count += 1
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

    def response_selection(self, fused_input, batch_size):
        response_select = self.selection_head(fused_input[:, :, 0].view(4 * batch_size, self.hidden_size))
        rs_loss = self.ce(response_select, torch.arange(4).to(fused_input.device).repeat(batch_size))
        return rs_loss

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
