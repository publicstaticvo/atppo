from util import *
from torch import nn
from models import ATForTPP, WavLMMAMHead
from transformers import PreTrainedModel


class ActorModel(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_unexpected = [r"mlm_head", r"mam_head", r"selection_head", r"masked_spec_embed", r"position_ids", r"mask_token"]
    _keys_to_ignore_on_save = [r"prediction_head"]
    supports_gradient_checkpointing = True

    def __init__(self, config: ATConfig, *args, **kwargs):
        super(ActorModel, self).__init__(config)
        self.model = ATForTPP(config=config)
        self.hidden_size = config.text.hidden_size
        self.num_ends = config.fused.num_ends
        self.start_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))
        self.end_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))
        self.vocab_size = config.text.vocab_size

    def predict(self, text, mask):
        words = text.masked_select(mask.unsqueeze(-1)).view(-1, self.hidden_size)
        return self.start_prediction_head(words).squeeze(-1)

    def forward(self, audio_input, text_input, audio_attention_mask, text_attention_mask, mlm_label=None,
                turn_id=None, start_valid=None, end_valid=None):
        # start_valid和end_valid是8轮对话concat到一起的真实valid，no_grad模式下大小为[Lt](1:B)，grad模式下大小为B * Lt。
        fused_features, mam_label, a_masked = self.model(audio_input, text_input, audio_attention_mask, text_attention_mask, turn_id, mlm_label is not None)
        bs, text_len = mlm_label.shape
        fused_features = fused_features.view(bs, 4, -1, self.hidden_size)
        text_fused = fused_features[:, 0, :text_len]
        if isinstance(start_valid, list):
            pred_start, pred_end = [], []
            for t, s, e in zip(text_fused, start_valid, end_valid):
                pred_start.append(self.predict(t, s))
                pred_end.append(self.predict(t, e))
            return pred_start, pred_end, fused_features, text_fused, mam_label, a_masked
        pred_start = self.predict(text_fused, start_valid)
        pred_end = self.predict(text_fused, end_valid)
        return pred_start, pred_end, fused_features, text_fused, mam_label, a_masked
