import torch
import torch.nn as nn
from typing import Optional, Tuple
from ._transformers import MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from .transformer_modules.RoPE import PositionalEmbedding
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.Attention import (  # noqa F401
    IPEXTransformerAttnNaive,
    IPEXTransformerAttnOptimizedFp16,
)  # noqa
from .transformer_modules.QuantizedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedInt4,
)  # noqa
from .transformer_modules.Mlp import *  # noqa
from .transformer_modules.DecoderBlock import IPEXTransformerBlock
from .transformer_modules.Linear import (  # noqa F401
    IPEXTransformerLinear,
)  # noqa
from .transformer_modules.XPUAttentionfp16 import (
    IPEXAttention,
)
from .transformer_modules.XPUAttentionInt4 import (
    IPEXAttentionInt4,
    IPEXAttentionInt4OneDNN,
)


class NewIPEXBloomBlock(IPEXTransformerBlock):
    def __init__(
        self,
        module,
        config: IPEXTransformerConfig,
        dtype="fp16",
        device="xpu",
        module_name="",
        impl_mode=None,
        tp_size=1,
        tp_group=None,
        **kwargs,
    ):
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(
            config, device, dtype, impl_mode, tp_size, tp_group
        )
        if dtype == "fp16":
            self.attn = IPEXAttention(self.ipex_config)
        elif dtype == "int4" and xpu_gemm_use_xetla():
            self.attn = IPEXAttentionInt4(self.ipex_config)
        elif dtype == "int4" and not xpu_gemm_use_xetla():
            self.attn = IPEXAttentionInt4OneDNN(self.ipex_config)
        else:
            raise NotImplementedError(
                "IPEXAttention dose not support this modelType {} !".format(dtype)
            )
        self.mlp = self.build_mlp_from_config()
        self.input_layernorm = nn.LayerNorm(
            self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
        )
        self.post_attention_layernorm = nn.LayerNorm(
            self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
        )
        self.port_all_parameters_to_new_module()

    def build_ipex_transformer_config(
        self, config, device, dtype, impl_mode, tp_size, tp_group
    ):
        activation_function = "bloom_gelu"
        ipex_activation = None
        for act in SupportedActivation:
            if activation_function in act.value:
                ipex_activation = act
                break
        assert ipex_activation is not None, (
            "found unrecognized activation function, can"
            "not build ipex config from {}".format(activation_function)
        )
        assert dtype in [
            "fp16",
            "int4",
        ], "dtype tag {} passed to optimized_transformers" "is not supported!".format(
            dtype
        )

        ipex_config = IPEXTransformerConfig(
            embedding_dim=self.config.hidden_size,
            intermediate_dim=4 * self.config.hidden_size,
            num_attention_head=self.config.n_head,
            num_key_value_head=self.config.n_head,
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=PositionalEmbedding,
            rotary_dim=None,
            use_causal_mask=True,
            activation_function=activation_function,
            ipex_act=ipex_activation,
            norm_eps=self.config.layer_norm_epsilon,
            residual_dropout=None,
            attn_dropout=self.config.attention_dropout,
            enable_bias=False,
            residual_pdrop=None,
            scale_attention=True,
            is_decoder=True,
            do_norm_before=self.config.apply_residual_connection_post_layernorm,
            ln_elementwise_affine=None,
            dtype=dtype,
            impl=impl_mode,
            tp_size=tp_size,
            tp_group=tp_group,
        )
        ipex_config.max_positions = max(ipex_config.max_positions, MAX_SEQ_LEN)
        return ipex_config

    def port_attn_parameter(self):
        embed_dim = self.ipex_config.embedding_dim
        num_head = self.ipex_config.num_attention_head // self.ipex_config.tp_size
        weight_shape = [num_head, 3, -1, embed_dim]
        bias_shape = [num_head, 3, -1]
        qkv_weight = self.module.self_attention.query_key_value.weight
        qkv_bias = self.module.self_attention.query_key_value.bias
        qkv_weight.data = (
            qkv_weight.view(weight_shape)
            .transpose(0, 1)
            .contiguous()
            .view([3, -1, embed_dim])
            .contiguous()
        )
        qkv_bias.data = (
            qkv_bias.view(bias_shape)
            .transpose(0, 1)
            .contiguous()
            .view([3, -1])
            .contiguous()
        )
        q_proj = IPEXTransformerLinear(qkv_weight[0], qkv_bias[0])
        k_proj = IPEXTransformerLinear(qkv_weight[1], qkv_bias[1])
        v_proj = IPEXTransformerLinear(qkv_weight[2], qkv_bias[2])
        self.attn.load_parameter(
            q_proj, k_proj, v_proj, self.module.self_attention.dense
        )

    def port_mlp_parameter(self):
        self.mlp.load_parameter(
            self.module.mlp.dense_h_to_4h, self.module.mlp.dense_4h_to_h
        )

    def port_norm_parameter(self):
        self.input_layernorm.weight = self.module.input_layernorm.weight
        self.input_layernorm.bias = self.module.input_layernorm.bias
        self.post_attention_layernorm.weight = (
            self.module.post_attention_layernorm.weight
        )
        self.post_attention_layernorm.bias = self.module.post_attention_layernorm.bias

    def transpose_parameter(self):
        self.attn.transpose_parameter()
        self.mlp.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()
        self.attn.cat_qkv()
        self.module.self_attention.query_key_value.weight.data = (
            self.attn.qkv_proj.weight.data
        )
        self.module.self_attention.query_key_value.bias.data = (
            self.attn.qkv_proj.bias.data
        )

    def release_resources(self):
        self.attn.release_resources()
        self.mlp.release_resources()

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        bs = IPEXTransformerAttn.batch_size
        IPEXTransformerAttn.beam_size = hidden_states.shape[0] // bs
        layernorm_output = torch.ops.torch_ipex.fast_layer_norm(
            hidden_states,
            self.input_layernorm.normalized_shape,
            self.input_layernorm.weight,
            self.input_layernorm.bias,
            self.input_layernorm.eps,
        )
        if self.ipex_config.do_norm_before:
            residual = layernorm_output
        else:
            residual = hidden_states
        attn_outputs = self.attn(
            hidden_states=layernorm_output,
            past_key_value=layer_past,
            attention_mask=None,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            residual=residual,
            alibi=alibi,
            cache_position=cache_position,
        )

        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        layernorm_output = torch.ops.torch_ipex.fast_layer_norm(
            attention_output,
            self.post_attention_layernorm.normalized_shape,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.bias,
            self.post_attention_layernorm.eps,
        )
        if self.ipex_config.do_norm_before:
            residual = layernorm_output
        else:
            residual = attention_output

        hidden_states = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs
