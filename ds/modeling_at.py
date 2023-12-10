import os
import sys
from torch import nn
from transformers.activations import ACT2FN
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from model import ATModel


class ATForSequenceClassification(nn.Module):
    def __init__(self, ckpt_path, config, num_class):
        super().__init__()
        self.model = ATModel.from_pretrained(ckpt_path, config=config, train=False)
        self.num_class = num_class
        hidden_size = config.text.hidden_size
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), ACT2FN['gelu'], nn.Linear(hidden_size, self.num_class))
        self.config = config
        if self.num_class == 1:
            self.loss_fct = nn.L1Loss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, audio_input, text_input, audio_attention_mask, text_attention_mask, turn_ids=None, labels=None):
        fused_features = self.model(audio_input, text_input, audio_attention_mask, text_attention_mask, turn_ids=turn_ids)
        logits = self.head(fused_features).squeeze(1)
        if labels is None:
            return logits
        loss = self.loss_fct(logits, labels.to(dtype=logits.dtype) if self.num_class == 1 else labels)
        return logits, loss
