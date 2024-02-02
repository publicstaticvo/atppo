from util import *
from .modeling_at import ATMultiTurnModel


class AT23062682(ATMultiTurnModel):

    def __init__(self, config: ATConfig, audio=None, text=None, *args, **kwargs):
        super(AT23062682, self).__init__(config, audio, text)

    def get_fused_input(self, audio_features, audio_mask, text_features, text_mask):
        bs = audio_features.shape[0]
        text_len = text_features.shape[:1]
        text_samples = text_features.shape[0] // bs
        token_type_ids = torch.zeros([bs * text_samples, text_len + audio_features.shape[1]], dtype=torch.long).to(text_features.device)
        token_type_ids[:, text_len:] = 1
        audio_features = audio_features.unsqueeze(1).repeat(1, text_samples, 1, 1).view(bs * text_samples, -1, self.hidden_size)
        fused_input = torch.cat([text_features, audio_features], dim=1) + self.token_type_embeddings(token_type_ids)
        audio_mask = audio_mask.unsqueeze(1).repeat(1, text_samples, 1).view(bs * text_samples, -1)
        fused_attention_mask = torch.cat([text_mask, audio_mask], dim=1).to(dtype=text_features.dtype)
        fused_attention_mask = (1.0 - fused_attention_mask[:, None, None, :]) * torch.finfo(text_features.dtype).min
        return fused_input, fused_attention_mask
