import torch
from typing import Optional, Tuple, Union
from functools import partial

from .transformer_modules.RoPE import Phi3RotaryEmbedding
from .transformer_modules.Norm import LlamaRMSNorm

Phi3RMSNorm = LlamaRMSNorm

from ._transformers import MAX_OUT_SEQ_LEN
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
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.Mlp import *  # noqa
from .transformer_modules.QuantizedMlp import *  # noqa
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from .transformer_modules.DecoderBlock import IPEXTransformerBlock
from .transformer_modules.model_utils import (
    load_attn_fused_qkv_params,
    transpose_attn_fused_qkv_params,
)


class NewIPEXPhi3DecoderLayer(IPEXTransformerBlock):
    def __init__(
        self,
        module,
        config,
        layer_idx,
        dtype="fp16",
        device="xpu",
        module_name="",
        impl_mode=None,
        tp_size=1,
        tp_group=None,
    ):
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(
            config, device, dtype, impl_mode, tp_size, tp_group
        )
        self.layer_idx = layer_idx
        if dtype == "fp16":
            self.attn = IPEXAttention(self.ipex_config, self.layer_idx)
        elif dtype == "int4" and xpu_gemm_use_xetla():
            self.attn = IPEXAttentionInt4(self.ipex_config, module.self_attn.layer_idx)
        elif dtype == "int4" and not xpu_gemm_use_xetla():
            self.attn = IPEXAttentionInt4OneDNN(
                self.ipex_config, module.self_attn.layer_idx
            )
        else:
            raise NotImplementedError(
                "IPEXAttention dose not support this modelType {} !".format(dtype)
            )
        # self.attn.post_qkv = partial(phi3_post_qkv, self.attn)
        self.attn.position_embed = self.ipex_config.rotary_embedding_class(
            self.ipex_config, torch.float16
        )
        self.mlp = self.build_mlp_from_config("phi3")
        self.input_layernorm = Phi3RMSNorm(
            self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
        )
        self.post_attention_layernorm = Phi3RMSNorm(
            self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
        )
        self.port_all_parameters_to_new_module()

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
            num_key_value_head=self.config.num_key_value_heads,
            max_positions=self.config.max_position_embeddings,
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=Phi3RotaryEmbedding,
            # rotary_dim=self.config.rotary_dim,
            use_causal_mask=False,
            activation_function=activation_function,
            ipex_act=ipex_activation,
            norm_eps=self.config.rms_norm_eps,
            resid_pdrop=self.config.resid_pdrop,
            attn_dropout=self.config.attention_dropout,
            scale_attention=True,
            sliding_window=self.config.sliding_window,
            dtype=dtype,
            impl=impl_mode,
            tp_size=tp_size,
            tp_group=tp_group,
        )

    def port_attn_parameter(self):
        self.attn.load_parameter(
            self.module.self_attn.qkv_proj,
            self.module.self_attn.o_proj,
            dtype=self.ipex_config.dtype,
        )

    def port_mlp_parameter(self):
        self.mlp.load_parameter(self.module.mlp.gate_up_proj, self.module.mlp.down_proj)

    def port_norm_parameter(self):
        self.input_layernorm.weight = self.module.input_layernorm.weight
        self.post_attention_layernorm.weight = (
            self.module.post_attention_layernorm.weight
        )

    def transpose_parameter(self):
        self.attn.transpose_parameter(dtype=self.ipex_config.dtype)
        self.mlp.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        self.attn.load_parameter = partial(load_attn_fused_qkv_params, self.attn)
        self.attn.transpose_parameter = partial(
            transpose_attn_fused_qkv_params, self.attn
        )
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        # hidden_states:  [bs*beam, seq, hidden_size]
        # position_ids:   [bs*beam, seq]
        # attention_mask: [bs*beam, head, q_seq, kv_seq]
        bs = IPEXTransformerAttn.batch_size
        IPEXTransformerAttn.beam_size = hidden_states.shape[0] // bs

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, self_attn_weights = self.attn(
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
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs
