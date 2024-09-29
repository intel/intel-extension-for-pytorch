import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from .transformer_modules.Attention import IPEXTransformerAttnOptimizedFp16  # noqa

from .transformer_modules.RoPE import PositionalEmbedding

from ._transformers import MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.Mlp import (  # noqa F401
    IPEXTransformerBaseMLP,
    IPEXTransformerMLPOptimizedFp16,
)  # noqa
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from .transformer_modules.QuantizedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedInt4,
)  # noqa
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .transformer_modules.CrossedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16Crossed,
)  # noqa
from .transformer_modules.DecoderBlock import IPEXTransformerBlock
from .transformer_modules.Mlp import *  # noqa
from .transformer_modules.XPUAttentionfp16 import (
    IPEXAttention,
)
from .transformer_modules.XPUAttentionInt4 import (
    IPEXAttentionInt4,
    IPEXAttentionInt4OneDNN,
)


class NewIPEXOPTBlock(IPEXTransformerBlock):
    def __init__(
        self,
        module,
        config,
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
            self.self_attn = IPEXAttention(self.ipex_config)
        elif dtype == "int4" and xpu_gemm_use_xetla():
            self.self_attn = IPEXAttentionInt4(self.ipex_config)
        elif dtype == "int4" and not xpu_gemm_use_xetla():
            self.self_attn = IPEXAttentionInt4OneDNN(self.ipex_config)
        else:
            raise NotImplementedError(
                "IPEXAttention dose not support this modelType {} !".format(dtype)
            )
        self.mlp = self.build_mlp_from_config("Opt")
        self.do_layer_norm_before = self.ipex_config.do_norm_before
        self.self_attn_layer_norm = nn.LayerNorm(
            self.ipex_config.embedding_dim,
            elementwise_affine=self.ipex_config.ln_elementwise_affine,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.ipex_config.embedding_dim,
            elementwise_affine=self.ipex_config.ln_elementwise_affine,
        )
        self.dropout_p = self.ipex_config.residual_pdrop
        self.port_all_parameters_to_new_module()

    def build_ipex_transformer_config(
        self, config, device, dtype, impl_mode, tp_size, tp_group
    ) -> IPEXTransformerConfig:
        activation_function = self.config.activation_function
        ipex_activation = None
        for act in SupportedActivation:
            if activation_function in act.value:
                ipex_activation = act
                break
        assert ipex_activation is not None, (
            "found unrecognized activation "
            "function, can not build ipex config from {}".format(activation_function)
        )

        assert dtype in [
            "fp16",
            "int4",
        ], "dtype tag {} passed to " "optimized_transformers is not supported!".format(
            dtype
        )

        return IPEXTransformerConfig(
            embedding_dim=self.config.hidden_size,
            intermediate_dim=self.config.ffn_dim,
            num_attention_head=self.config.num_attention_heads,
            num_key_value_head=self.config.num_attention_heads,
            max_positions=max(self.config.max_position_embeddings, MAX_SEQ_LEN),
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=PositionalEmbedding,
            rotary_dim=None,
            rotary_half=False,
            rotate_every_two=False,
            use_causal_mask=False,
            activation_function=self.config.activation_function,
            ipex_act=ipex_activation,
            norm_eps=None,
            residual_dropout=self.config.dropout,
            attn_dropout=None,
            enable_bias=self.config.enable_bias,
            residual_pdrop=self.config.dropout,
            scale_attention=True,
            is_decoder=True,
            do_norm_before=self.config.do_layer_norm_before,
            ln_elementwise_affine=self.config.layer_norm_elementwise_affine,
            positional_embedding_base=10000,
            device=self.device,
            dtype=dtype,
            impl=impl_mode,
            tp_size=tp_size,
            tp_group=tp_group,
        )

    def port_attn_parameter(self):
        # IPEXTransformerAttnOptimizedFp16Opt IPEXTransformerAttnOptimizedFp16
        self.self_attn.load_parameter(
            self.module.self_attn.q_proj,
            self.module.self_attn.k_proj,
            self.module.self_attn.v_proj,
            self.module.self_attn.out_proj,
        )

    def port_mlp_parameter(self):
        # IPEXTransformerMLPOptimizedFp16ReluOpt IPEXTransformerAttnOptimizedFp16
        self.mlp.load_parameter(self.module.fc1, self.module.fc2)

    def port_norm_parameter(self):
        self.self_attn_layer_norm.weight = self.module.self_attn_layer_norm.weight
        self.self_attn_layer_norm.bias = self.module.self_attn_layer_norm.bias
        self.final_layer_norm.weight = self.module.final_layer_norm.weight
        self.final_layer_norm.bias = self.module.final_layer_norm.bias

    def transpose_parameter(self):
        self.self_attn.transpose_parameter()
        self.mlp.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()
        self.self_attn.cat_qkv()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[List[torch.FloatTensor]] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        bs = IPEXTransformerAttn.batch_size
        IPEXTransformerAttn.beam_size = hidden_states.shape[0] // bs

        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = torch.ops.torch_ipex.fast_layer_norm(
                hidden_states,
                self.self_attn_layer_norm.normalized_shape,
                self.self_attn_layer_norm.weight,
                self.self_attn_layer_norm.bias,
                self.self_attn_layer_norm.eps,
            )

        hidden_states, present_key_value, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            head_mask=layer_head_mask,
            output_attentions=output_attentions,
            residual=residual,
            past_key_value=past_key_value,
        )

        if not self.do_layer_norm_before:
            hidden_states = torch.ops.torch_ipex.fast_layer_norm(
                hidden_states,
                self.self_attn_layer_norm.normalized_shape,
                self.self_attn_layer_norm.weight,
                self.self_attn_layer_norm.bias,
                self.self_attn_layer_norm.eps,
            )

        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = torch.ops.torch_ipex.fast_layer_norm(
                hidden_states,
                self.final_layer_norm.normalized_shape,
                self.final_layer_norm.weight,
                self.final_layer_norm.bias,
                self.final_layer_norm.eps,
            )
        hidden_states = self.mlp(hidden_states, residual)
        if not self.do_layer_norm_before:
            hidden_states = torch.ops.torch_ipex.fast_layer_norm(
                hidden_states,
                self.final_layer_norm.normalized_shape,
                self.final_layer_norm.weight,
                self.final_layer_norm.bias,
                self.final_layer_norm.eps,
            )

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs
