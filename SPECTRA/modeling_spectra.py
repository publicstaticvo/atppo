from util import *
from models import WavLMForCRS, ATModel


class SpectraModel(ATModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_missing = ["mlm_head", "mam_head", "selection_head", "start_prediction_head", "end_prediction_head", r"position_ids", r"mask_token"]
    _keys_to_ignore_on_load_unexpected = [r"masked_spec_embed"]
    supports_gradient_checkpointing = True

    def __init__(self, config: ATConfig, audio=None, text=None, *args, **kwargs):
        super(SpectraModel, self).__init__(config, audio_class=WavLMForCRS, audio=audio, text=text)

    def fuse_four(self, text, audio, bs, text_len, audio_len, token_type_ids=None):
        text = text.unsqueeze(2).repeat(1, 1, 2, 1, 1).view(4 * bs, text_len, -1)
        audio = audio.unsqueeze(1).repeat(1, 2, 1, 1, 1).view(4 * bs, audio_len, -1)
        fused_input = torch.cat([text, audio], dim=1)
        if token_type_ids is not None:
            fused_input += self.token_type_embeddings(token_type_ids).half()
        else:
            fused_input = fused_input.squeeze(-1)
        return fused_input

    def get_fused_input(self, audio_features, audio_mask, text_features, text_mask):
        bs = text_features.shape[0] // 2
        text_len = text_features.shape[1]
        audio_features = audio_features.view(bs, 2, -1, self.hidden_size)
        text_features = text_features.view(bs, 2, text_len, self.hidden_size)
        audio_len = audio_features.shape[2]
        token_type_ids = torch.zeros([bs * 4, text_len + audio_len], dtype=torch.long).to(audio_mask.device)
        token_type_ids[:, text_len:] = 1
        fused_input = self.fuse_four(text_features, audio_features, bs, text_len, audio_len, token_type_ids)
        fused_attention_mask = self.fuse_four(text_mask, audio_mask, bs, text_len, audio_len).to(dtype=text_features.dtype)
        fused_attention_mask = (1.0 - fused_attention_mask[:, None, None, :]) * torch.finfo(text_features.dtype).min
        return fused_input, fused_attention_mask

    def forward(self, audio_input, text_input, audio_mask=None, text_mask=None, turn_id=None, mask_modeling=False, *args, **kwargs):
        out = self.audio_encoder(audio_input, audio_mask, perform_mam=mask_modeling, token_embedding=self.text_encoder.embeddings.token_type_embeddings)
        audio_features, audio_mask = out[:2]
        text_features = self.text_encoder(text_input, text_mask, token_type_ids=turn_id)[0]
        fused_input, fused_attention_mask = self.get_fused_input(audio_features, audio_mask, text_features, text_mask)
        fused_input = self.fused_encoder(fused_input, fused_attention_mask).last_hidden_state
        return fused_input, out[2], out[3]
