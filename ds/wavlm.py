import math
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers import WavLMConfig


class WavLMNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(self.in_conv_dim, self.out_conv_dim, kernel_size=config.conv_kernel[layer_id],
                              stride=config.conv_stride[layer_id], bias=config.conv_bias)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class WavLMLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(self.in_conv_dim, self.out_conv_dim, kernel_size=config.conv_kernel[layer_id],
                              stride=config.conv_stride[layer_id], bias=config.conv_bias)
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class WavLMGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(self.in_conv_dim, self.out_conv_dim, kernel_size=config.conv_kernel[layer_id],
                              stride=config.conv_stride[layer_id], bias=config.conv_bias)
        self.activation = ACT2FN[config.feat_extract_activation]
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class WavLMPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=config.num_conv_pos_embeddings,
                              padding=config.num_conv_pos_embeddings // 2, groups=config.num_conv_pos_embedding_groups)
        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
        self.padding = WavLMSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class WavLMSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class WavLMFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.feat_extract_norm == "group":
            conv_layers = [WavLMGroupNormConvLayer(config, layer_id=0)]
            for i in range(1, config.num_feat_extract_layers - 1):
                conv_layers.append(WavLMNoLayerNormConvLayer(config, layer_id=i))
            if config.last_conv_layer == "no":
                conv_layers.append(WavLMNoLayerNormConvLayer(config, layer_id=config.num_feat_extract_layers - 1))
            elif config.last_conv_layer == "layer":
                conv_layers.append(WavLMLayerNormConvLayer(config, layer_id=config.num_feat_extract_layers - 1))
            else:
                conv_layers.append(WavLMGroupNormConvLayer(config, layer_id=config.num_feat_extract_layers - 1))
        elif config.feat_extract_norm == "layer":
            conv_layers = [WavLMLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            raise ValueError(f"config.feat_extract_norm {config.feat_extract_norm} has to be one of ['group', 'layer']")
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def forward(self, input_values):
        hidden_states = input_values[:, None]
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True
        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(conv_layer), hidden_states)
            else:
                hidden_states = conv_layer(hidden_states)
        return hidden_states


class WavLMFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class WavLMAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, num_buckets=320, max_distance=800,
                 has_relative_position_bias=True, ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)

        if has_relative_position_bias:
            self.rel_attn_embed = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False):
        bsz, tgt_len, _ = hidden_states.size()
        if position_bias is None:
            position_bias = self.compute_bias(tgt_len, tgt_len)
            position_bias = (
                position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, tgt_len)
            )
        gated_hidden_states = hidden_states.view(hidden_states.shape[:-1] + (self.num_heads, -1))
        gated_hidden_states = gated_hidden_states.permute(0, 2, 1, 3)
        relative_position_proj = self.gru_rel_pos_linear(gated_hidden_states)
        relative_position_proj = relative_position_proj.view(gated_hidden_states.shape[:-1] + (2, 4)).sum(-1)
        gate_a, gate_b = torch.sigmoid(relative_position_proj).chunk(2, dim=-1)
        gate_output = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
        gated_position_bias = gate_output.view(bsz * self.num_heads, -1, 1) * position_bias
        gated_position_bias = gated_position_bias.view((-1, tgt_len, tgt_len))
        attn_output, attn_weights = self.torch_multi_head_self_attention(
            hidden_states, attention_mask, gated_position_bias, output_attentions)
        return attn_output, attn_weights, position_bias

    def torch_multi_head_self_attention(self, hidden_states, attention_mask, gated_position_bias, output_attentions):
        query = key = value = hidden_states.transpose(0, 1)
        key_padding_mask = attention_mask.ne(1) if attention_mask is not None else None
        bias_k = bias_v = None
        add_zero_attn = False
        attn_output, attn_weights = nn.functional.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads, torch.empty([0]),
            torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            bias_k, bias_v, add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, self.training,
            key_padding_mask, output_attentions, gated_position_bias, use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight)
        attn_output = attn_output.transpose(0, 1)
        if attn_weights is not None:
            attn_weights = attn_weights[:, None].broadcast_to(
                attn_weights.shape[:1] + (self.num_heads,) + attn_weights.shape[1:]
            )
        return attn_output, attn_weights

    def compute_bias(self, query_length: int, key_length: int) -> torch.FloatTensor:
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position)
        relative_position_bucket = relative_position_bucket.to(self.rel_attn_embed.weight.device)
        values = self.rel_attn_embed(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def _relative_positions_bucket(self, relative_positions: torch.FloatTensor) -> torch.FloatTensor:
        num_buckets = self.num_buckets // 2
        relative_buckets = (relative_positions > 0).to(torch.long) * num_buckets
        relative_positions = torch.abs(relative_positions)
        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact
        relative_positions_if_large = torch.log(relative_positions.float() / max_exact)
        relative_positions_if_large = relative_positions_if_large / math.log(self.max_distance / max_exact)
        relative_positions_if_large = relative_positions_if_large * (num_buckets - max_exact)
        relative_position_if_large = (max_exact + relative_positions_if_large).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_positions, relative_position_if_large)
        return relative_buckets


class WavLMFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class WavLMEncoderLayer(nn.Module):
    def __init__(self, config: WavLMConfig, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = WavLMAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads,
                                        dropout=config.attention_dropout, num_buckets=config.num_buckets,
                                        max_distance=config.max_bucket_distance,
                                        has_relative_position_bias=has_relative_position_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias = self.attention(hidden_states, attention_mask=attention_mask,
                                                                    position_bias=position_bias,
                                                                    output_attentions=output_attentions)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states, position_bias)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class WavLMEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config: WavLMConfig, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = WavLMAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads,
                                        dropout=config.attention_dropout, num_buckets=config.num_buckets,
                                        max_distance=config.max_bucket_distance,
                                        has_relative_position_bias=has_relative_position_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, position_bias = self.attention(hidden_states, attention_mask=attention_mask,
                                                                    position_bias=position_bias,
                                                                    output_attentions=output_attentions)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))
        outputs = (hidden_states, position_bias)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class WavLMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = WavLMPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [WavLMEncoderLayer(config, has_relative_position_bias=(i == 0)) for i in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, output_hidden_states=False,
                return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if attention_mask is not None:
            hidden_states[~attention_mask] = 0.0
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        position_bias = None
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            dropout_probability = np.random.uniform(0, 1)
            skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer), hidden_states,
                                                                      attention_mask, position_bias)
                else:
                    layer_outputs = layer(hidden_states, attention_mask=attention_mask, position_bias=position_bias,
                                          output_attentions=output_attentions)
                hidden_states, position_bias = layer_outputs[:2]
            if skip_the_layer:
                layer_outputs = (None, None)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class WavLMEncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = WavLMPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([WavLMEncoderLayerStableLayerNorm(config, has_relative_position_bias=(i == 0))
                                     for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, output_hidden_states=False,
                return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if attention_mask is not None:
            # make sure padded tokens are not attended to
            hidden_states[~attention_mask] = 0
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        position_bias = None
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            dropout_probability = np.random.uniform(0, 1)
            skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer), hidden_states,
                                                                      attention_mask, position_bias)
                else:
                    layer_outputs = layer(hidden_states, attention_mask=attention_mask, position_bias=position_bias,
                                          output_attentions=output_attentions, index=i)
                hidden_states, position_bias = layer_outputs[:2]
            if skip_the_layer:
                layer_outputs = (None, None)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)


class WavLMAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # feature dim might need to be down-projected
        if config.output_hidden_size != config.hidden_size:
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            self.proj = self.proj_layer_norm = None

        self.layers = nn.ModuleList(WavLMAdapterLayer(config) for _ in range(config.num_adapter_layers))
        self.layerdrop = config.layerdrop

    def forward(self, hidden_states):
        # down project hidden_states if necessary
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)

        for layer in self.layers:
            layerdrop_prob = np.random.random()
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class WavLMAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(config.output_hidden_size, 2 * config.output_hidden_size, config.adapter_kernel_size,
                              stride=config.adapter_stride, padding=1)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)
        return hidden_states


class WavLMMAMHead(nn.Module):
    def __init__(self, hidden_size, output_dim, hidden_act: str = "gelu", dr=1, layer_norm_eps=1e-5):
        super().__init__()
        self.output_dim = output_dim
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act_fn = ACT2FN[hidden_act]
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, self.output_dim * dr, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.output_dim * dr))
        self.decoder.bias = self.bias

    def forward(self, features):
        x = self.dense(features)
        x = self.act_fn(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class WavLMPreTrainedModel(PreTrainedModel):
    config_class = WavLMConfig
    base_model_prefix = "wavlm"
    main_input_name = "input_values"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, WavLMPositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)))
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, WavLMFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(self, input_lengths, add_adapter=None):
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode='trunc') + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask, add_adapter=None):
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)
        batch_size = attention_mask.shape[0]
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return torch.clamp_min(output_lengths, 0).tolist(), attention_mask

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (WavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder)):
            module.gradient_checkpointing = value


class WavLMForMAM(WavLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"mask_token"]
    _keys_to_ignore_on_load_unexpected = [r"masked_spec_embed"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.feature_extractor = WavLMFeatureEncoder(config)
        self.feature_projection = WavLMFeatureProjection(config)
        if config.has_audio_cls:
            self.audio_cls = nn.Parameter(torch.randn(config.hidden_size), requires_grad=True)
            if config.multi_turn:
                self.audio_sep = nn.Parameter(torch.randn(config.hidden_size), requires_grad=True)
        if config.do_stable_layer_norm:
            self.encoder = WavLMEncoderStableLayerNorm(config)
        else:
            self.encoder = WavLMEncoder(config)
        self.adapter = WavLMAdapter(config) if config.add_adapter else None
        self.post_init()

    def forward(self, input_values, attention_mask, turn_embeddings=None):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        out_len, attention_mask = self._get_feature_vector_attention_mask(
            extract_features.shape[1], attention_mask, add_adapter=False)
        hidden_states, extract_features = self.feature_projection(extract_features)
        if self.config.multi_turn:
            audio_len = hidden_states.shape[1]
            new_len = audio_len * 2 + 2
            bs = hidden_states.shape[0] // 2
            a1, a2 = torch.split(hidden_states.view(bs, 2, -1, self.config.hidden_size), 1, dim=1)
            m1, m2 = torch.split(attention_mask.view(bs, 2, -1), 1, dim=1)
            a1, a2, m1, m2 = map(lambda x: x.squeeze(1), [a1, a2, m1, m2])
            hidden_states = torch.zeros([bs, new_len, self.config.hidden_size], device=a1.device, dtype=a1.dtype)
            attention_mask = torch.zeros([bs, new_len], device=a1.device, dtype=torch.bool)
            token_type_id = torch.ones([bs, new_len], device=a1.device, dtype=torch.long)
            for i in range(bs):
                la1, la2 = out_len[2 * i: 2 * i + 2]
                pl = la1 + la2 + 2
                hidden_states[i, :pl] = torch.cat([self.audio_cls.unsqueeze(0), a1[i, :la1], self.audio_sep.unsqueeze(0), a2[i, :la2]], dim=0)
                token_type_id[i, :la1 + 1] = 0
                attention_mask[i, :pl] = True
        else:
            if self.config.has_audio_cls:
                bs = hidden_states.shape[0]
                attention_mask = torch.cat([torch.zeros(bs, 1).long().to(input_values.device), attention_mask], dim=1)
                hidden_states = torch.cat([self.audio_cls[None, None, :].repeat(bs, 1, 1), hidden_states], dim=1)
            token_type_id = torch.ones(hidden_states.shape[:2], dtype=torch.long, device=input_values.device)
        if turn_embeddings is not None:
            hidden_states += turn_embeddings(token_type_id)
        hidden_states = self.encoder(hidden_states, attention_mask, self.config.output_attentions,
                                     self.config.output_hidden_states, self.config.use_return_dict)[0]
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)
        return hidden_states, attention_mask
