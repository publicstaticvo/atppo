from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from models import ATSingleTurnModel, ATMultiTurnModel, WavLMForMultiTurn, WavLMForMultiModal
from util import ATConfig


class ATForSequenceClassification(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_missing = ["head"]
    supports_gradient_checkpointing = True

    def __init__(self, config: ATConfig, task, num_class, *model_args, **model_kwargs):
        super(ATForSequenceClassification, self).__init__(config)
        model_class = ATMultiTurnModel if "ic" in task else ATSingleTurnModel
        self.model = model_class(config)
        self.num_class = num_class
        hidden_size = config.text.hidden_size
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), ACT2FN['gelu'], nn.Linear(hidden_size, self.num_class))
        self.config = config
        if self.num_class == 1:
            self.loss_fct = nn.L1Loss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, audio_input, text_input, audio_attention_mask, text_attention_mask, turn_id=None, labels=None):
        fused_features, *_ = self.model(audio_input, text_input, audio_attention_mask, text_attention_mask, turn_id=turn_id)
        logits = self.head(fused_features[:, 0]).squeeze(1)
        if labels is None:
            return logits
        loss = self.loss_fct(logits, labels.to(dtype=logits.dtype) if self.num_class == 1 else labels)
        return logits, loss
