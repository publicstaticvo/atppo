from util import *
from torch import nn
from transformers import PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaLMHead
from .wavlm import WavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder, WavLMMAMHead


class TrainerBase(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_missing = ["mlm_head", "mam_head", "selection_head"]
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (RobertaEncoder, WavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder)):
            module.gradient_checkpointing = value

    def __init__(self, config):
        super(TrainerBase, self).__init__(config)
        self.vocab_size = config.text.vocab_size
        self.hidden_size = config.text.hidden_size
        self.mlm_head = RobertaLMHead(config.text)
        self.mam_head = WavLMMAMHead(self.hidden_size, config.audio.conv_dim[-1])
        self.selection_head = nn.Linear(self.hidden_size, 4)
        self.ce = torch.nn.CrossEntropyLoss()
        self.l1 = torch.nn.L1Loss()

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

    def forward(self, *args, **kwargs):
        raise NotImplementedError
