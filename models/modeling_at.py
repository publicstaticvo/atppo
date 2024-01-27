from util import *
from torch import nn
from .wavlm import WavLMForMultiModal, WavLMForMultiTurn
from transformers import PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaEncoder


class ATModel(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_missing = ["mlm_head", "mam_head", "selection_head", "start_prediction_head", "end_prediction_head", r"position_ids", r"mask_token"]
    # _keys_to_ignore_on_save = ["mlm_head", "mam_head", "selection_head", "start_prediction_head", "end_prediction_head"]
    _keys_to_ignore_on_load_unexpected = [r"masked_spec_embed"]
    supports_gradient_checkpointing = True

    def __init__(self, config: ATConfig, audio_class, audio=None, text=None, *args, **kwargs):
        super(ATModel, self).__init__(config)
        self.hidden_size = config.text.hidden_size
        self.num_fused_layers = config.fused.num_hidden_layers

        if audio is None:
            self.audio_encoder = audio_class(config.audio)
            self.text_encoder = RobertaModel(config.text)
        else:
            self.audio_encoder = audio_class.from_pretrained(audio, config=config.audio)
            self.text_encoder = RobertaModel.from_pretrained(text, config=config.text)
        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
        self.fused_encoder = RobertaEncoder(config.fused) if self.num_fused_layers > 0 else None

    def get_fused_input(self, audio_features, audio_mask, text_features, text_mask):
        bs, text_len = text_features.shape[:2]
        token_type_ids = torch.zeros([bs, text_len + audio_features.shape[1]], dtype=torch.long).to(text_features.device)
        token_type_ids[:, text_len:] = 1
        fused_input = torch.cat([text_features, audio_features], dim=1) + self.token_type_embeddings(token_type_ids)
        fused_attention_mask = torch.cat([text_mask, audio_mask], dim=1).to(dtype=text_features.dtype)
        fused_attention_mask = (1.0 - fused_attention_mask[:, None, None, :]) * torch.finfo(text_features.dtype).min
        return fused_input, fused_attention_mask

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class ATSingleTurnModel(ATModel):

    def __init__(self, config: ATConfig, audio=None, text=None, *args, **kwargs):
        super(ATSingleTurnModel, self).__init__(config, WavLMForMultiModal, audio, text)

    def forward(self, audio_input, text_input, audio_mask=None, text_mask=None, *args, **kwargs):
        audio_features, audio_mask, *_ = self.audio_encoder(audio_input, audio_mask)
        text_features = self.text_encoder(text_input, text_mask)[0]
        if self.num_fused_layers > 0:
            fused_input, fused_attention_mask = self.get_fused_input(audio_features, audio_mask, text_features, text_mask)
            fused_input = self.fused_encoder(fused_input, fused_attention_mask).last_hidden_state
            text_len = text_input.shape[1]
            text_features = fused_input[:, :text_len]
            audio_features = fused_input[:, text_len:]
        return text_features, audio_features, audio_mask


class ATMultiTurnModel(ATModel):

    def __init__(self, config: ATConfig, audio=None, text=None, *args, **kwargs):
        super(ATMultiTurnModel, self).__init__(config, WavLMForMultiTurn, audio, text)

    def forward(self, audio_input, text_input, audio_mask=None, text_mask=None, turn_id=None, head_mask_for_fused=None,
                mask_modeling=False, output_attentions=False, *args, **kwargs):
        audio_features, audio_mask, mam_labels, a_masked = self.audio_encoder(audio_input, audio_mask, perform_mam=mask_modeling, token_embedding=self.text_encoder.embeddings.token_type_embeddings)
        text_features = self.text_encoder(text_input, text_mask, token_type_ids=turn_id)[0]
        if self.num_fused_layers > 0:
            fused_input, fused_attention_mask = self.get_fused_input(audio_features, audio_mask, text_features, text_mask)
            fused_output = self.fused_encoder(fused_input, fused_attention_mask, output_attentions=output_attentions, head_mask=head_mask_for_fused)
            fused_features = fused_output.last_hidden_state
            if output_attentions:
                attention = fused_output.all_self_attentions
                return (fused_features, attention), mam_labels, a_masked
            return fused_features, mam_labels, a_masked
        return (audio_features, audio_mask, text_features), mam_labels, a_masked
