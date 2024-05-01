from util import *
from torch import nn
from models import WavLMMAMHead
from SPECTRA import SpectraModel
from transformers import PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead


class ActorModel(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_unexpected = [r"mlm_head", r"mam_head", r"selection_head", r"masked_spec_embed", r"position_ids", r"mask_token"]
    _keys_to_ignore_on_save = [r"prediction_head"]
    supports_gradient_checkpointing = True

    def __init__(self, config: ATConfig, perform_mlm=False, *args, **kwargs):
        super(ActorModel, self).__init__(config)
        self.model = SpectraModel(config=config)
        self.num_ends = config.fused.num_ends
        self.vocab_size = config.text.vocab_size
        self.hidden_size = config.text.hidden_size
        self.start_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))
        self.end_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, self.num_ends))
        if perform_mlm:
            self.mlm_head = RobertaLMHead(self.config.text)
            self.mam_head = WavLMMAMHead(self.hidden_size, self.config.audio.conv_dim[-1])
            self.selection_head = nn.Linear(self.hidden_size, 4)
        self.ce = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()

    def predict(self, text, mask, head):
        words = text.masked_select(mask.unsqueeze(-1)).view(-1, self.hidden_size)
        return head(words).squeeze(-1)

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

    def forward(self, audio_input, text_input, audio_attention_mask, text_attention_mask, mlm_label=None,
                turn_id=None, start_valid=None, end_valid=None, perform_mlm=False):
        # start_valid和end_valid是8轮对话concat到一起的真实valid，no_grad模式下大小为[Lt](1:B)，grad模式下大小为B * Lt。
        fused_features, mam_label, a_masked = self.model(audio_input, text_input, audio_attention_mask, text_attention_mask, turn_id, mlm_label is not None)
        bs, text_len = start_valid.shape
        fused_features = fused_features.view(bs, 4, -1, self.hidden_size)
        text_fused = fused_features[:, 0, :text_len]
        pred_start = self.predict(text_fused, start_valid, self.start_prediction_head)
        pred_end = self.predict(text_fused, end_valid, self.end_prediction_head)
        if perform_mlm:
            mlm = self.mlm_loss(text_fused, mlm_label)
            mam = self.mam_loss(fused_features[:, 0, text_len:], mam_label, a_masked)
            rs_loss = self.response_selection(fused_features, bs)
            return pred_start, pred_end, mlm, mam, rs_loss
        return pred_start, pred_end
