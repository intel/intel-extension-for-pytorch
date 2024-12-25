import torch
from typing import Optional, Tuple
from .transformer_modules.RoPE import LlamaRotaryEmbedding
from .transformer_modules.Norm import LlamaRMSNorm


from ._transformers import MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from .transformer_modules.XPUAttentionfp16 import (
    IPEXAttention,
)
from .transformer_modules.XPUAttentionInt4 import (
    IPEXAttentionInt4,
    IPEXAttentionInt4OneDNN,
)
from .transformer_modules.QuantizedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16,
    IPEXTransformerAttnOptimizedInt4,
)  # noqa
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .transformer_modules.GroupedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16Grouped,
    IPEXTransformerAttnOptimizedInt4Grouped,
)
from .transformer_modules.DecoderBlock import IPEXTransformerBlock
from .transformer_modules.Mlp import *  # noqa
from .transformer_modules.QuantizedMlp import *  # noqa

import os

acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in [
    "1",
    "ON",
    "Y",
    "YES",
    "TRUE",
]


class NewIPEXMistralBlock(IPEXTransformerBlock):
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
        grouped = False
        if self.ipex_config.num_attention_head > self.ipex_config.num_key_value_head:
            grouped = True
        # self.self_attn = self.build_attention_from_config(grouped=grouped)
        if dtype == "fp16":
            self.self_attn = IPEXAttention(self.ipex_config, module.self_attn.layer_idx)
        elif dtype == "int4" and xpu_gemm_use_xetla():
            self.self_attn = IPEXAttentionInt4(
                self.ipex_config, module.self_attn.layer_idx
            )
        elif dtype == "int4" and not xpu_gemm_use_xetla():
            self.self_attn = IPEXAttentionInt4OneDNN(
                self.ipex_config, module.self_attn.layer_idx
            )
        else:
            raise NotImplementedError(
                "IPEXAttention dose not support this modelType {} !".format(dtype)
            )
        self.mlp = self.build_mlp_from_config("llama")
        self.input_layernorm = LlamaRMSNorm(
            self.ipex_config.embedding_dim, self.ipex_config.norm_eps
        )
        self.post_attn_layernorm = LlamaRMSNorm(
            self.ipex_config.embedding_dim, self.ipex_config.norm_eps
        )
        self.port_all_parameters_to_new_module()
        self.layer_idx = kwargs.get("layer_idx", None)

    def build_ipex_transformer_config(
        self, config, device, dtype, impl_mode, tp_size, tp_group
    ) -> IPEXTransformerConfig:
        activation_function = self.config.hidden_act
        ipex_activation = None
        for act in SupportedActivation:
            if activation_function in act.value:
                ipex_activation = act
                break
        assert ipex_activation is not None, (
            "found unrecognized activation function,"
            "can not build ipex config from {}".format(activation_function)
        )

        assert dtype in [
            "fp16",
            "int4",
        ], "dtype tag {} passed to optimized_transformers is not supported!".format(
            dtype
        )

        return IPEXTransformerConfig(
            embedding_dim=self.config.hidden_size,
            intermediate_dim=self.config.intermediate_size,
            num_attention_head=self.config.num_attention_heads,
            # transformers==4.31.0
            num_key_value_head=self.config.num_key_value_heads,
            max_positions=max(self.config.max_position_embeddings, MAX_SEQ_LEN),
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=LlamaRotaryEmbedding,
            rotary_dim=None,
            rotary_half=True,
            rotate_every_two=False,
            use_causal_mask=False,
            activation_function=self.config.hidden_act,
            ipex_act=ipex_activation,
            norm_eps=self.config.rms_norm_eps,
            residual_dropout=None,
            attn_dropout=None,
            enable_bias=False,
            residual_pdrop=None,
            scale_attention=True,
            is_decoder=False,
            do_norm_before=None,
            ln_elementwise_affine=None,
            positional_embedding_base=self.config.rope_theta,
            device=self.device,
            dtype=dtype,
            impl=impl_mode,
            tp_size=tp_size,
            tp_group=tp_group,
        )

    def port_attn_parameter(self):
        self.self_attn.load_parameter(
            self.module.self_attn.q_proj,
            self.module.self_attn.k_proj,
            self.module.self_attn.v_proj,
            self.module.self_attn.o_proj,
        )

    def port_mlp_parameter(self):
        # IPEXTransformerMLPOptimizedFp16SiluLlama
        self.mlp.load_parameter(
            self.module.mlp.gate_proj,
            self.module.mlp.down_proj,
            self.module.mlp.up_proj,
        )

    def port_norm_parameter(self):
        self.input_layernorm.weight = self.module.input_layernorm.weight
        self.post_attn_layernorm.weight = self.module.post_attention_layernorm.weight

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
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        bs = IPEXTransformerAttn.batch_size
        IPEXTransformerAttn.beam_size = hidden_states.shape[0] // bs

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            residual=residual,
        )

        residual = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (past_key_value,)
        return outputs
