from util import ATConfig
from rm_trainer import AlignTrainer
from models import ATSingleTurnForSentenceAlign


class SentenceAlignTrainer(AlignTrainer):

    def __init__(self, config: ATConfig, audio=None, text=None, bias=False, maximum_reward=5):
        super(SentenceAlignTrainer, self).__init__(config, ATSingleTurnForSentenceAlign, audio, text, bias=bias)
        self.M = maximum_reward

    def reward_loss(self, scores, eps=1e-3):
        # 除此之外还可以.sigmoid()
        scores = scores.clamp(-self.M, self.M).exp().view(-1, self.model.num_items_per_sample)
        scores_cumsum = scores.flip([-1]).cumsum(-1).flip([-1])[:, :-1]
        return -(scores[:, :-1] / scores_cumsum.clamp_min(eps)).log().mean()

    def forward(self, audio_input, text_input, audio_mask, text_mask):
        scores = self.model(audio_input, text_input, audio_mask, text_mask)
        pro_loss = self.reward_loss(scores)
        return pro_loss
