import torch
import torch.nn as nn
from typing import Optional, Tuple

from intel_extension_for_pytorch.nn.utils._transformer_configuration import IPEXTransformerConfig
from ._transformers import IPEXTransformerAtten, IPEXTransformerMLP, IPEXEmptyLinear
from ._transformer_configuration import IPEXTransformerConfig
from ._transformer_converter import IPEXTransformerConverter, MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from .RoPE import PositionalEmbedding
import math

class IPEXBloomAttn(IPEXTransformerAtten):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.query_key_value = IPEXEmptyLinear()
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

    def qkv_normal(self, hidden_states, layer_past = None):
        if self.row_major:
            shape = [hidden_states.shape[0], hidden_states.shape[1], self.num_attn_head * self.head_dim]
            query = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            key = torch.empty_like(query)
            value = torch.empty_like(query)
            torch.ops.torch_ipex.mm_qkv_out(hidden_states, self.qkv_wei, self.qkv_bias, query, key, value)
        else:
            fused_qkv = self.query_key_value(hidden_states)
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_attn_head, 3, self.head_dim)
            query = fused_qkv[..., 0, :].reshape(batch_size, seq_length, -1)
            key = fused_qkv[..., 1, :].reshape(batch_size, seq_length, -1)
            value = fused_qkv[..., 2, :].reshape(batch_size, seq_length, -1)
        return query, key, value


class IPEXBloomMLP(IPEXTransformerMLP):
    def __init__(self, config: IPEXTransformerConfig):
        super().__init__(config)

    def forward(self, hidden_states, residual: torch.Tensor):
        if self.row_major:
            if isinstance(self.act, nn.GELU):
                hidden_states = torch.ops.torch_ipex.matmul_gelu(hidden_states, self.fc_in_wei, self.fc_in.bias, self.act.approximate)
            else:
                hidden_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.fc_in_wei, self.fc_in.bias)
                hidden_states = self.act(hidden_states)
            hidden_states = torch.ops.torch_ipex.mm_bias_resadd(hidden_states, self.fc_out_wei, self.fc_out_bias, residual, 1.0/self.tp_size)
            output = self.all_reduce_if_necessary(hidden_states)
        else:
            intermediate_output = self.act(self.fc_in(hidden_states))
            output = torch.matmul(intermediate_output, self.fc_out.weight.t())
            output = self.all_reduce_if_necessary(output)
            output += self.fc_out.bias + residual
        return output


class IPEXBloomBlock(nn.Module):
    def __init__(self, 
                 config: IPEXTransformerConfig):
        super().__init__()
        self.config = config
        self.input_layernorm = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)
        self.self_attention = IPEXBloomAttn(config)
        self.post_attention_layernorm = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)
        self.mlp = IPEXBloomMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # convert layout form [beam, seq, hidden_size] to [seq, beam, hidden_size]
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        #layernorm_output = self.input_layernorm(hidden_states)
        layernorm_output = torch.ops.torch_ipex.fast_layer_norm(hidden_states, self.input_layernorm.normalized_shape, self.input_layernorm.weight, self.input_layernorm.bias, self.input_layernorm.eps)
        if self.config.do_norm_before:
            residual = layernorm_output
        else:
            residual = hidden_states
        attn_outputs = self.self_attention(
            hidden_states = layernorm_output,
            layer_past = layer_past,
            attention_mask = attention_mask,
            head_mask = head_mask,
            use_cache = use_cache,
            output_attentions = output_attentions,
            residual=residual,
            alibi=alibi
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]
        #layernorm_output = self.post_attention_layernorm(attention_output)
        layernorm_output = torch.ops.torch_ipex.fast_layer_norm(attention_output, self.post_attention_layernorm.normalized_shape, self.post_attention_layernorm.weight, self.post_attention_layernorm.bias, self.post_attention_layernorm.eps)
        if self.config.do_norm_before:
            redisual = layernorm_output
        else:
            residual = attention_output

        output = self.mlp(layernorm_output, residual)
        output = output.transpose(0, 1)

        if use_cache:
            outputs = (output, ) + outputs
        else:
            outputs = (output, ) + outputs[1:]
        return outputs


def _convert_to_bloom_cache_ipex(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, seq_length, head_dim = past_key_value[0][0].shape

        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, seq_length, head_dim),
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )

class IPEXBloomConverter(IPEXTransformerConverter):
    def __init__(self, module, config, device="cpu", dtype=torch.float) -> None:
        from transformers.models.bloom.configuration_bloom import BloomConfig
        super().__init__(module, config, device, dtype)
        self.config: BloomConfig = config if config is not None else BloomConfig()
        self.ipex_transformers_config = self.construct_transformer_config()
        self.ipex_optimized_module = self.construct_ipex_optimized_module()
        self.port_all_parameters_to_new_module()

    def construct_transformer_config(self):
        # bloom don't have n_position attribute, we set it as 2048 just like other LLM models.
        n_positions = max(2048, MAX_SEQ_LEN)
        embed_dim = self.config.hidden_size
        num_head = self.config.n_head
        hidden_dropout = self.config.hidden_dropout
        attention_dropout = self.config.attention_dropout
        before_norm = self.config.apply_residual_connection_post_layernorm
        # activate_function = self.config.hidden_act
        norm_eps = self.config.layer_norm_epsilon
        use_cache = self.config.use_cache
        intermediate_size = 4 * embed_dim
        return IPEXTransformerConfig(
            embed_dim=embed_dim,
            intermediate_size=intermediate_size,
            num_attention_heads=num_head,
            max_positions=n_positions,
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=PositionalEmbedding,
            rotary_dim=None,
            rotate_half=False,
            rotate_every_two=False,
            use_casual_mask=False,
            activation_function="bloom_gelu",
            norm_eps=norm_eps,
            residual_dropout=None,
            attn_dropout=attention_dropout,
            enable_bias=False,
            residual_pdrop=None,
            scale_attention=True,
            is_decoder=False,
            do_norm_before=before_norm,
            ln_elementwise_affine=None,
            seq_first=True,
            kv_cache_optimize=True,
            positional_embedding_base=10000,
            sdp_fusion_enable=True,
            device=self.device,
            dtype=self.dtype,
            tp_size=IPEXTransformerConverter.tp_size,
            tp_group=IPEXTransformerConverter.tp_group
        )
    def construct_ipex_optimized_module(self):
        return IPEXBloomBlock(self.ipex_transformers_config)

    def port_attn_parameters(self):
        tp_size = self.ipex_transformers_config.tp_size
        if self.row_major:
            embed_dim = self.config.hidden_size
            num_head = self.config.n_head // tp_size
            shape = [num_head, 3, -1, embed_dim]

            self.module.self_attention.query_key_value.weight.data = \
                self.module.self_attention.query_key_value.weight.view(shape).contiguous().transpose(0, 1).contiguous().view([3, -1, embed_dim]).transpose(1, 2).contiguous()
            self.ipex_optimized_module.self_attention.qkv_wei = \
                self.module.self_attention.query_key_value.weight

            self.module.self_attention.query_key_value.bias.data = \
                self.module.self_attention.query_key_value.bias.view([num_head, 3, -1]).transpose(0, 1).contiguous().view([3, -1])
            self.ipex_optimized_module.self_attention.qkv_bias = \
                self.module.self_attention.query_key_value.bias

            self.module.self_attention.dense.weight.data = \
                self.module.self_attention.dense.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.self_attention.out_wei = self.module.self_attention.dense.weight
            self.module.self_attention.dense.bias.data = self.module.self_attention.dense.bias.data / self.tp_size
            self.ipex_optimized_module.self_attention.out_bias = self.module.self_attention.dense.bias
        else:
            self.ipex_optimized_module.self_attention.query_key_value.weight = nn.Parameter(self.module.self_attention.query_key_value.weight)
            self.ipex_optimized_module.self_attention.query_key_value.bias = nn.Parameter(self.module.self_attention.query_key_value.bias)
            self.ipex_optimized_module.self_attention.out_proj.weight = nn.Parameter(self.module.self_attention.dense.weight)
            self.ipex_optimized_module.self_attention.out_proj.bias = nn.Parameter(self.module.self_attention.dense.bias)


    def port_mlp_parameters(self):
        if self.row_major:
            self.module.mlp.dense_h_to_4h.weight.data = self.module.mlp.dense_h_to_4h.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_in_wei = self.module.mlp.dense_h_to_4h.weight
            self.ipex_optimized_module.mlp.fc_in.bias = nn.Parameter(self.module.mlp.dense_h_to_4h.bias)

            self.module.mlp.dense_4h_to_h.weight.data = self.module.mlp.dense_4h_to_h.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_out_wei = self.module.mlp.dense_4h_to_h.weight
            self.module.mlp.dense_4h_to_h.bias.data = self.module.mlp.dense_4h_to_h.bias.data / self.tp_size
            self.ipex_optimized_module.mlp.fc_out_bias = self.module.mlp.dense_4h_to_h.bias
        else:
            self.ipex_optimized_module.mlp.fc_in.weight = nn.Parameter(self.module.mlp.dense_h_to_4h.weight)
            self.ipex_optimized_module.mlp.fc_in.bias = nn.Parameter(self.module.mlp.dense_h_to_4h.bias)
            self.ipex_optimized_module.mlp.fc_out.weight = nn.Parameter(self.module.mlp.dense_4h_to_h.weight)
            self.ipex_optimized_module.mlp.fc_out.bias = nn.Parameter(self.module.mlp.dense_4h_to_h.bias)

    def port_layer_norm_parameters(self):
        self.ipex_optimized_module.input_layernorm.weight = nn.Parameter(self.module.input_layernorm.weight)
        self.ipex_optimized_module.input_layernorm.bias = nn.Parameter(self.module.input_layernorm.bias)
        self.ipex_optimized_module.post_attention_layernorm.weight = nn.Parameter(self.module.post_attention_layernorm.weight)
        self.ipex_optimized_module.post_attention_layernorm.bias = nn.Parameter(self.module.post_attention_layernorm.bias)

    def port_all_parameters_to_new_module(self):
        self.port_attn_parameters()
        self.port_mlp_parameters()
        self.port_layer_norm_parameters()

    def get_transformed_module(self):
        return self.ipex_optimized_module
