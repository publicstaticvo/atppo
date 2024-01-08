import torch
from util import ATConfig
from models import WavLMMAMHead
from transformers import PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead


class AlignTrainer(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_unexpected = [r"fused_encoder"]

    def __init__(self, config: ATConfig, model_class, audio=None, text=None):
        super(AlignTrainer, self).__init__(config)
        self.hidden_size = config.text.hidden_size
        self.model = model_class(config, audio, text)
        if hasattr(config, "perform_mlm"):
            self.num_negative = config.num_negative
            self.perform_mlm = config.perform_mlm
            if self.perform_mlm:
                self.mlm_head = RobertaLMHead(config.text)
                self.mam_head = WavLMMAMHead(self.hidden_size, config.audio.conv_dim[-1])
                self.vocab_size = config.text.vocab_size
                self.ce = torch.nn.CrossEntropyLoss()
                self.l1 = torch.nn.L1Loss()
        self.temperature = 1

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

    def valid_filter(self, outputs, valid, pooling_mode):
        words = valid.shape[0]
        if pooling_mode == "first":
            # 每个valid形状为L
            return outputs.masked_select(valid.unsqueeze(-1)).view(-1, self.hidden_size)
        elif pooling_mode == "mean":
            # 每个valid形状为Ni*L
            temp = outputs.unsqueeze(0).repeat(words, 1, 1).masked_fill(~valid.unsqueeze(-1), 0)
            return torch.sum(temp, dim=1) / torch.sum(valid, dim=1, keepdim=True)

    def save_pretrained(self, save_directory, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
