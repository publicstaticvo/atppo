from util import *
from torch import nn
from models import WavLMMAMHead
from SPECTRA import SpectraModel
from transformers import PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead


class DPEngine(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_unexpected = [r"mlm_head", r"mam_head", r"selection_head", r"masked_spec_embed", r"position_ids", r"mask_token"]
    _keys_to_ignore_on_save = [r"prediction_head"]
    supports_gradient_checkpointing = True

    def __init__(self, config: ATConfig, perform_mlm=False, *args, **kwargs):
        super(DPEngine, self).__init__(config)
        self.model = SpectraModel(config=config, *args, **kwargs)
        self.num_ends = config.fused.num_ends
        self.vocab_size = config.text.vocab_size
        self.hidden_size = config.text.hidden_size
        self.duration_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))
        self.audio_to_word_head = RobertaLMHead(self.config.text)
        if perform_mlm:
            self.mlm_head = RobertaLMHead(self.config.text)
            self.mam_head = WavLMMAMHead(self.hidden_size, self.config.audio.conv_dim[-1])
            self.selection_head = nn.Linear(self.hidden_size, 4)
        self.ce = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()

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
            mam_loss = self.l1(mam_pre.masked_select(masked_indices[0].unsqueeze(-1)), label.masked_select(masked_indices[1]))
        return mam_loss

    def response_selection(self, fused_input, batch_size):
        response_select = self.selection_head(fused_input[:, :, 0].view(4 * batch_size, self.hidden_size))
        rs_loss = self.ce(response_select, torch.arange(4).to(fused_input.device).repeat(batch_size))
        return rs_loss

    def forward(self, audio_input, text_input, audio_attention_mask, text_attention_mask, mlm_label=None,
                turn_id=None, valid_filter=None, stage=1, reconstruct_label=None):
        # stage: 指示调用方法。
        # valid_filter是8轮对话concat到一起的真实valid，no_grad模式下大小为[Lt](1:B)，grad模式下大小为B * Lt。
        if stage == 1:
            fused_features, mam_label, a_masked, audio_mask = self.model(audio_input, text_input, audio_attention_mask,
                                                                         text_attention_mask, turn_id, True, True)
            bs, text_len = valid_filter.shape
            fused_features = fused_features.view(bs, 4, -1, self.hidden_size)
            text_fused = fused_features[:, 0, :text_len]
            mlm = self.mlm_loss(text_fused, mlm_label)
            mam = self.mam_loss(fused_features[:, 0, text_len:], mam_label, a_masked)
            rs_loss = self.response_selection(fused_features, bs)
            return mlm, mam, rs_loss
        elif stage == 2:
            # no grad
            fused_features, mam_label, a_masked, audio_mask = self.model(audio_input, text_input, audio_attention_mask,
                                                                         text_attention_mask, turn_id, False, False)
            text_len = valid_filter.shape[-1]
            text_fused = fused_features[:, :text_len]
            words = text_fused.masked_select(valid_filter.unsqueeze(-1)).view(-1, self.hidden_size)
            duration = self.duration_prediction_head(words).squeeze(-1)
            return duration, audio_mask
        elif stage == 3:
            with torch.no_grad():
                fused_features, mam_label, a_masked, audio_mask = self.model(audio_input, text_input, audio_attention_mask,
                                                                             text_attention_mask, turn_id, False, False)
            text_len = valid_filter.shape[-1]
            audio_fused = fused_features[:, text_len:]
            predict_words = self.audio_to_word_head(audio_fused)
            if reconstruct_label is None:
                return predict_words
            loss = self.ce(predict_words.view(-1, self.vocab_size), reconstruct_label.view(-1))
            return loss
