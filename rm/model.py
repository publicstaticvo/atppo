from torch import nn
from wavlm import WavLMForMultiTurn
from configuration_at import ATConfig
from transformers import PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaLayer, RobertaEncoder


class ATModel(PreTrainedModel):
    config_class = ATConfig
    # _keys_to_ignore_on_save = ["mlm_head", "mam_head", "selection_head", "start_prediction_head", "end_prediction_head"]
    supports_gradient_checkpointing = True

    def __init__(self, config: ATConfig, audio=None, text=None, tpp=True):
        super(ATModel, self).__init__(config)
        self.perform_mlm = config.perform_mlm
        self.tpp = tpp

        if audio is None:
            self.audio_encoder = WavLMForMultiTurn(config.audio)
            self.text_encoder = RobertaModel(config.text)
        else:
            self.audio_encoder = WavLMForMultiTurn.from_pretrained(audio, config=config.audio)
            self.text_encoder = RobertaModel.from_pretrained(text, config=config.text)
        self.token_type_embeddings = nn.Embedding(2, config.text.hidden_size)

    def forward(self, audio_input, text_input, audio_attention_mask=None, text_attention_mask=None, turn_id=None, perform_mam=False):
        audio_features, audio_mask, mam_label, a_masked = self.audio_encoder(audio_input, audio_attention_mask, perform_mam=perform_mam, token_embedding=self.text_encoder.embeddings.token_type_embeddings)
        text_features = self.text_encoder(text_input, text_attention_mask, token_type_ids=turn_id)[0]
        return audio_features, text_features, mam_label, a_masked
