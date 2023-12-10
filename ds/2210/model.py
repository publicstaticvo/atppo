import random
import torch
from torch import nn
from utils import ATConfig
from wavlm import WavLMForMAM
from torch.nn.functional import normalize
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaLayer


class SPCL:
    def __init__(self, num_classes, pool_size, num_shots, temperature):
        self.pool = [[] for _ in range(num_classes)]
        self.pool_size = pool_size
        self.num_shots = num_shots
        self.temperature = temperature

    def set_pool(self, pool):
        self.pool = [random.sample(p, self.pool_size) if len(p) > self.pool_size else p for p in pool]

    def __call__(self, logits, labels):
        with torch.no_grad():
            la = labels.detach().tolist()
            for i in range(logits.shape[0]):
                label = la[i]
                logit = logits[i]
                self.pool[label].append(logit.detach())
                if len(self.pool[label]) > self.pool_size:
                    self.pool[label] = self.pool[label][-self.pool_size:]
            can = []
            for p in self.pool:
                xp = torch.stack(random.sample(p, self.num_shots) if len(p) > self.num_shots else p, dim=0)
                can.append(xp)
            centers = torch.stack([torch.mean(x, dim=0) for x in can], dim=0)
        logits = normalize(logits, dim=-1)
        sim = torch.mm(logits, torch.cat([logits, normalize(centers, dim=-1)]).transpose(0, 1)) / self.temperature
        labels = labels.unsqueeze(1) == torch.cat([labels, torch.arange(len(self.pool)).to(sim.device)]).unsqueeze(0)
        # labels = labels ^ torch.eye(len(self.pool)).to(sim.device)
        sim = torch.softmax(sim, dim=-1)
        return torch.mean(torch.sum(labels * sim, dim=-1))


class ATModel(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [r"masked_spec_embed"]

    def __init__(self, config: ATConfig, audio=None, text=None):
        super(ATModel, self).__init__(config)
        self.hidden_size = config.text.hidden_size
        if audio is None:
            self.audio_encoder = WavLMForMAM(config.audio)
            self.text_encoder = RobertaModel(config.text)
        else:
            self.audio_encoder = WavLMForMAM.from_pretrained(audio, config=config.audio)
            self.text_encoder = RobertaModel.from_pretrained(text, config=config.text)
        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
        if hasattr(config.text, "num_fusion_layers"):
            self.fused_encoder = nn.ModuleList(RobertaLayer(config.text) for _ in range(config.text.num_fusion_layers))
        elif not hasattr(config.text, "no_fusion_layer"):
            self.fused_encoder = RobertaLayer(config.text)
        self.vocab_size = config.text.vocab_size
        self.conv_dim = config.audio.conv_dim[-1]

    def forward(self, audio_input, text_input, audio_attention_mask=None, text_attention_mask=None, turn_ids=None):
        audio_features, audio_mask = self.audio_encoder(audio_input, audio_attention_mask, turn_embeddings=self.text_encoder.embeddings.token_type_embeddings if self.config.audio.use_turn_ids else None)
        text_features = self.text_encoder(text_input, text_attention_mask, token_type_ids=turn_ids)[0]
        if hasattr(self.config.text, "no_fusion_layer"):
            return text_features
        bs, text_len = text_features.shape[:2]
        token_type_ids = torch.zeros([bs, text_len + audio_features.shape[1]], dtype=torch.long).to(text_input.device)
        token_type_ids[:, text_len:] = 1
        fused_input = torch.cat([text_features, audio_features], dim=1) + self.token_type_embeddings(token_type_ids)
        fused_attention_mask = torch.cat([text_attention_mask, audio_mask], dim=1).to(dtype=text_features.dtype)
        fused_attention_mask = (1.0 - fused_attention_mask[:, None, None, :]) * torch.finfo(text_features.dtype).min
        if hasattr(self.config.text, "num_fusion_layers"):
            for layer in self.fused_encoder:
                fused_input = layer(fused_input, fused_attention_mask)[0]
            return fused_input
        else:
            return self.fused_encoder(fused_input, fused_attention_mask)[0]


class DownstreamModel(nn.Module):
    def __init__(self, ckpt_path, config, num_class, turn_embeddings=None):
        super().__init__()
        self.model = ATModel.from_pretrained(ckpt_path, config=config)
        if turn_embeddings is not None:
            self.model.text_encoder.embeddings.token_type_embeddings.weight = nn.Parameter(turn_embeddings, requires_grad=True)
        self.num_class = num_class
        hidden_size = config.text.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            ACT2FN['gelu'],
            nn.Linear(hidden_size, self.num_class))
        self.config = config
        if self.num_class == 1:
            self.loss_fct = nn.L1Loss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        self.spcl = SPCL(num_class, config.pool_size, config.num_shots, config.temperature)

    def generate_representations(self, audio_input, text_input, a_mask, t_mask, turn_ids=None, prompt=None):
        fused_features = self.model(audio_input, text_input, a_mask, t_mask, turn_ids=turn_ids)
        if prompt is None:
            return fused_features[:, 0]
        return fused_features[torch.arange(fused_features.shape[0]).to(audio_input.device), prompt]

    def calc_loss(self, fused, labels, mode):
        logits = self.head(fused).squeeze(1)
        loss = self.loss_fct(logits, labels.to(dtype=logits.dtype) if self.num_class == 1 else labels)
        if mode == "train":
            return loss, self.spcl(fused, labels)
        return logits, loss

    def forward(self, audio_input, text_input, audio_attention_mask, text_attention_mask,
                turn_ids=None, labels=None, prompt=None, mode="train"):
        fused_features = self.generate_representations(audio_input, text_input, audio_attention_mask,
                                                       text_attention_mask, turn_ids, prompt)
        if labels is None:
            return self.head(fused_features).squeeze(1)
        return self.calc_loss(fused_features, labels, mode)

