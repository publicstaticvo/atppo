import torch
from util import ATConfig
from rm_trainer import AlignTrainer
from models import ATModelForSentenceAlign


class SentenceAlignTrainer(AlignTrainer):

    def __init__(self, config: ATConfig, audio=None, text=None, bias=False):
        super(SentenceAlignTrainer, self).__init__(config, ATModelForSentenceAlign, audio, text)
        self.reward_head = torch.nn.Linear(self.hidden_size, 1, bias=bias)

    def reward_loss(self, scores, eps=1e-3):
        scores = scores.exp().view(-1, self.model.num_items_per_sample)
        scores_cumsum = scores.flip([-1]).cumsum(-1).flip([-1])[:, :-1]
        return (scores[:, :-1] / scores_cumsum.clamp_min(eps)).log().mean()

    def forward(self, audio_input, text_input, audio_mask, text_mask, turn_id):
        fused_features = self.model(audio_input, text_input, audio_mask, text_mask, turn_id)[:, 0]
        # fused_features: (N+1)B * 768
        scores = self.reward_head(fused_features).unsqueeze(-1)  # (N+1)B
        pro_loss = self.reward_loss(scores)
        return pro_loss
