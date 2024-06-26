from util import *
from models import ATSingleTurnModel


class ATSingleTurnForSentenceAlign(ATSingleTurnModel):

    def __init__(self, config: ATConfig, audio=None, text=None, bias=False, *args, **kwargs):
        super(ATSingleTurnForSentenceAlign, self).__init__(config, audio, text)
        self.reward_head = torch.nn.Linear(self.hidden_size, 1, bias=bias)
        self.num_items_per_sample = config.num_negative + 1 if hasattr(config, "num_negative") else 1

    def get_fused_input(self, audio_features, audio_mask, text_features, text_mask):
        bs, text_len = text_features.shape[:2]
        token_type_ids = torch.zeros([bs * self.num_items_per_sample, text_len + audio_features.shape[1]], dtype=torch.long).to(text_features.device)
        token_type_ids[:, text_len:] = 1
        if self.num_items_per_sample > 1:
            text_features = text_features.unsqueeze(1).repeat(1, self.num_items_per_sample, 1, 1).view(bs * self.num_items_per_sample, text_len, self.hidden_size)
            text_mask = text_mask.unsqueeze(1).repeat(1, self.num_items_per_sample, 1).view(bs * self.num_items_per_sample, text_len)
        fused_input = torch.cat([text_features, audio_features], dim=1) + self.token_type_embeddings(token_type_ids)
        fused_attention_mask = torch.cat([text_mask, audio_mask], dim=1).to(dtype=text_features.dtype)
        fused_attention_mask = (1.0 - fused_attention_mask[:, None, None, :]) * torch.finfo(text_features.dtype).min
        return fused_input, fused_attention_mask

    def forward(self, audio_input, text_input, audio_mask=None, text_mask=None, *args, **kwargs):
        audio_features, audio_mask, *_ = self.audio_encoder(audio_input, audio_mask)
        text_features = self.text_encoder(text_input, text_mask)[0]
        fused_input, fused_attention_mask = self.get_fused_input(audio_features, audio_mask, text_features, text_mask)
        fused_input = self.fused_encoder(fused_input, fused_attention_mask).last_hidden_state[:, 0]
        scores = self.reward_head(fused_input).squeeze(-1)  # (N+1)B
        return scores
