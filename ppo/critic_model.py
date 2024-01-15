from util import *
from models import ATModel, WavLMForMultiModal


class ATModelForWordAlign(ATModel):

    def __init__(self, config: ATConfig, audio=None, text=None):
        super(ATModelForWordAlign, self).__init__(config, WavLMForMultiModal, audio, text)


class ATModelForSentenceAlign(ATModel):

    def __init__(self, config: ATConfig, audio=None, text=None, bias=False):
        super(ATModelForSentenceAlign, self).__init__(config, WavLMForMultiTurn, audio, text)
        self.reward_head = torch.nn.Linear(self.hidden_size, 1, bias=bias)
        self.num_items_per_sample = config.num_negative + 1

    def get_fused_input(self, audio_features, audio_mask, text_features, text_mask):
        bs, text_len = text_features.shape[:2]
        token_type_ids = torch.zeros([bs * self.num_items_per_sample, text_len + audio_features.shape[1]], dtype=torch.long).to(text_features.device)
        token_type_ids[:, text_len:] = 1
        text_features = text_features.unsqueeze(1).repeat(1, self.num_items_per_sample, 1, 1).view(bs * self.num_items_per_sample, text_len, self.hidden_size)
        fused_input = torch.cat([text_features, audio_features], dim=1) + self.token_type_embeddings(token_type_ids)
        text_mask = text_mask.unsqueeze(1).repeat(1, self.num_items_per_sample, 1).view(bs * self.num_items_per_sample, text_len)
        fused_attention_mask = torch.cat([text_mask, audio_mask], dim=1).to(dtype=text_features.dtype)
        fused_attention_mask = (1.0 - fused_attention_mask[:, None, None, :]) * torch.finfo(text_features.dtype).min
        return fused_input, fused_attention_mask

    def forward(self, audio_input, text_input, audio_mask=None, text_mask=None, turn_id=None, *args, **kwargs):
        # audio: 2(N+1)B * 160000  text: B * L  mlm_label: B * L  turn_id: B * L
        audio_features, audio_mask, *_ = self.audio_encoder(audio_input, audio_mask, token_embedding=self.text_encoder.embeddings.token_type_embeddings)
        # audio_features: (N+1)B * 200 * 768  audio_mask: (N+1)B * 200
        text_features = self.text_encoder(text_input, text_mask, token_type_ids=turn_id)[0]
        # text_features: B * L * 768
        fused_input, fused_attention_mask = self.get_fused_input(audio_features, audio_mask, text_features, text_mask)
        fused_input = self.fused_encoder(fused_input, fused_attention_mask).last_hidden_state
        # fused_features: (N+1)B * 768
        scores = self.reward_head(fused_input).unsqueeze(-1)  # (N+1)B
        return scores
