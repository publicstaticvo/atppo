import json
import deepspeed
from apex import amp
from torch.optim import AdamW
from apex.optimizers import FusedAdam
from torch.nn.functional import normalize
from transformers import get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP

from util import *
from tpp.tpp_trainer import ATForTPP
from configuration_at import ATConfig
CONFIG = "config.json"


def step(loss, model, args, mode, optimizer=None, scheduler=None):
    if mode == "ds":
        model.backward(loss)
        model.step()
    else:
        if "amp" in mode:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if args.grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.grad_norm)
        else:
            loss.backward()
            if args.grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        if model.step_count % args.grad_acc == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


def split_audio_features(audio, audio_attention_mask):
    bs = audio_attention_mask.shape[0] // 2
    audio_attention_mask_sum = torch.clamp_max(torch.sum(audio_attention_mask.view(bs, 2, -1), dim=-1) // 320, 99)
    audio_attention_mask_sum[:, 0] += 1
    audio_attention_mask_sum = audio_attention_mask_sum.long().tolist()
    audio_features = []
    for i in range(bs):
        audio_features.append(audio[1:audio_attention_mask_sum[i][0]])
        audio_features.append(audio[audio_attention_mask_sum[i][0] + 1:])
        assert audio_features[-1].shape[0] == audio_attention_mask_sum[i][1]
    return audio_features


def split_text_words(text, split_mark):
    text_words = []
    for t, m in zip(text, split_mark):
        text_words.append(t[:m])
        text_words.append(t[m:])
    return text_words


class UnsupervisedTrainer:

    def __init__(self, args):
        self.generator, self.g_optim, self.g_scheduler, self.g_mode = self.init_generator(args)
        self.discriminator, self.d_optim, self.d_scheduler, self.d_mode = self.init_generator(args)

        self.args = args
        self.perform_mlm = True
        self.hidden_size = self.actor.module.config.text.hidden_size if hasattr(self.actor, "module") else self.actor.config.text.hidden_size
        self.kl_ctl = 0.1
        self.clip_reward_value = 10
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.experience_count = 0
        self.mean_map = construct_mean_map(200).half()

    def valid_filter(self, outputs, valid, pooling_mode):
        words = valid.shape[0]
        if pooling_mode == "first":
            # 每个valid形状为L
            return outputs.masked_select(valid.unsqueeze(-1)).view(-1, self.hidden_size)
        elif pooling_mode == "mean":
            # 每个valid形状为Ni*L
            temp = outputs.unsqueeze(0).repeat(words, 1, 1).masked_fill(~valid.unsqueeze(-1), 0)
            return torch.sum(temp, dim=1) / torch.sum(valid, dim=1, keepdim=True)

    def train_gan(self):
        # Train Discriminator with real images
        real_labels = torch.ones(images.size(0), 1)
        outputs = self.discriminator(images)
        d_loss_real = criterion(outputs, real_labels)
        # Train Discriminator with fake images
        z = torch.randn(images.size(0), 100)
        fake_images = self.generator(z)
        fake_labels = torch.zeros(images.size(0), 1)
        outputs = self.discriminator(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        # Backprop and optimize for discriminator
        d_loss = d_loss_real + d_loss_fake
        step(d_loss, self.discriminator, self.args, mode, self.d_optim, self.d_scheduler)
        # Train Generator
        z = torch.randn(images.size(0), 100)
        fake_images = self.generator(z)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        step(g_loss, self.generator, self.args, mode, self.g_optim, self.g_scheduler)

    def train_ppo(self, audio_input, text_input, audio_mask, text_mask, mlm_label=None, turn_id=None, start_valid=None, end_valid=None, splits=None):
        # 1 标注
        with torch.no_grad():
            tpp_starts, tpp_ends = self.compute_alignment(audio_input, text_input, audio_mask, text_mask, turn_id, start_valid, splits)
        # 2 求偏差，actor step
        fused_features, mam_label, a_masked = self.actor.model(audio_input, text_input, audio_mask, text_mask, turn_id, self.perform_mlm)
        bs, text_len = mlm_label.shape
        fused_features = fused_features.view(bs, 4, -1, self.hidden_size)
        text_fused = fused_features[:, 0, :text_len]
        start_valid, end_valid = map(lambda x: torch.cat(x), [start_valid, end_valid])
        mlm = self.actor.mlm_loss(text_fused, mlm_label)
        mam = self.actor.mam_loss(fused_features[:, 0, text_len:], mam_label, a_masked)
        rs_loss = self.actor.response_selection(fused_features, bs)
        span_loss, pred_actor = self.actor.tpp_loss(text_fused, start_valid, end_valid, tpp_starts, tpp_ends)
        with torch.no_grad():
            pred_ref = self.ref.tpp_loss(text_fused, start_valid, end_valid)
        kl = self.kl(pred_actor, pred_ref)
        loss = mlm + mam + rs_loss + span_loss * 3 + self.kl_ctl * kl
        step(loss, self.actor, self.args, self.actor_mode, self.actor_optim, self.actor_scheduler)
        return mlm, mam, rs_loss, span_loss, kl, loss

    def init_generator(self, args):
        # 1 ds config
        if args.model_path:
            config = ATConfig.from_pretrained(args.model_path)
        else:
            config = ATConfig.from_json_files(os.path.join(args.audio_path, CONFIG), os.path.join(args.text_path, CONFIG))
            config.set_length(int(args.audio_length * SAMPLE_RATE), args.text_length)
            config.fused.num_hidden_layers = args.num_fused_layers
            config.fused.num_ends = args.num_ends
        config.audio.train_mode = 3
        tokenizer = RobertaTokenizerFast.from_pretrained(args.text_path)
        # 4。读输入数据
        train_data = TPPDataset(args.transcripts, args.num_turns, args.file_prefix)
        # 5。整理config并建立模型
        if args.no_pretrain:
            model = ATForTPP(config)
        elif args.model_path:
            model = ATForTPP.from_pretrained(args.model_path, config=config)
        else:
            model = ATForTPP(config, args.audio_path, args.text_path)
        # 6。数据并行
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        no_decay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        ogp = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        num_train_steps = args.train_epochs * math.ceil(len(train_data) / args.batch_size / args.grad_acc)
        if args.apex_level > 0:
            from apex import amp
            from apex.optimizers import FusedAdam
            optimizer = FusedAdam(ogp, lr=args.lr, bias_correction=False)
        else:
            optimizer = AdamW(ogp, lr=args.lr, eps=1e-8)
        warmup_steps = int(args.warmup * num_train_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_steps)
        c = DataCollatorForTPP(tokenizer, config, args.apex_level > 0)
        # 4 parallel
        if args.ds_config:
            model, optimizer, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer, config=args.ds_config, lr_scheduler=scheduler, dist_init_required=True)
            model_mode = "ds"
        else:
            model.to(args.device)
            model_mode = ""
            if args.apex_level > 0:
                model, optimizer = amp.initialize(model, optimizer, opt_level=f"O{args.apex_level}", keep_batchnorm_fp32=False if args.apex_level >= 2 else None, loss_scale="dynamic" if args.loss_scale == 0. else args.loss_scale)
                model_mode = "amp"
            if args.local_rank >= 0:
                model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=[args.local_rank])
                model_mode += "ddp"
        return model, optimizer, scheduler, model_mode

    def init_ref(self, args, model_class, model_name):
        # 1 ds config
        if args.ds_config:
            args.ds_config = get_eval_ds_config(args.batch_size, args.ds_stage, args.apex_level)
        # 2 model
        model = model_class.from_pretrained(model_name)
        # 3 parallel
        if args.ds_config:
            model, *_ = deepspeed.initialize(model=model, config=args.ds_config, dist_init_required=True)
        else:
            model.to(args.device)
            if args.apex_level > 0:
                model = amp.initialize(model, opt_level=f"O{args.apex_level}", keep_batchnorm_fp32=False if args.apex_level >= 2 else None)
            if args.local_rank >= 0:
                model = DDP(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=[args.local_rank])
        return model

    def save_pretrained(self, path):
        temp = self.actor
        while hasattr(temp, "module"):
            temp = temp.module
        temp.save_pretrained(path)

    def train(self):
        self.actor.train()

    def eval(self):
        self.actor.eval()
        self.ref.eval()
        self.reward.eval()
