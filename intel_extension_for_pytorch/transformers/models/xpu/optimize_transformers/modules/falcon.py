import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from ._transformers import MAX_OUT_SEQ_LEN, MAX_SEQ_LEN
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.Decoderblock import IPEXTransformerBlock
from .transformer_modules.GroupedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16Grouped,
)
from .transformer_modules.Mlp import *  # noqa
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .transformer_modules.QuantizedAttention import (  # noqa F401; noqa
    IPEXTransformerAttnOptimizedFp16,
    IPEXTransformerAttnOptimizedInt4,
)
from .transformer_modules.RoPE import LlamaRotaryEmbedding

acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in [
    "1",
    "ON",
    "Y",
    "YES",
    "TRUE",
]


class NewIPEXFalconBlock(IPEXTransformerBlock):
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
    ):
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(
            config, device, dtype, impl_mode, tp_size, tp_group
        )
        self.attn = self.build_attention_from_config()
        self.mlp = self.build_mlp_from_config()
        self.ln_attn = nn.LayerNorm(
            self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
        )
        self.ln_mlp = nn.LayerNorm(
            self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
        )
        self.port_all_parameters_to_new_module()

    def build_attention_from_config(self):
        dtype = self.ipex_config.dtype
        impl = self.ipex_config.impl
        attn_type = IPEXTransformerAttn
        attn_type_str = "IPEXTransformerAttn"
        for elem in [impl.name, dtype, "Grouped"]:
            attn_type_str = attn_type_str + elem.capitalize()
            if hasattr(sys.modules[__name__], attn_type_str):
                attn_type = getattr(sys.modules[__name__], attn_type_str)
        attn = attn_type(self.ipex_config)
        attn.module = [self.module.self_attention]
        return attn

    def build_mlp_from_config(self):
        dtype = self.ipex_config.dtype
        impl = self.ipex_config.impl
        activation = self.ipex_config.ipex_act
        mlp_type = IPEXTransformerMLP
        mlp_type_str = "IPEXTransformerMLP"
        for elem in [impl.name, dtype, activation.name, "Falcon"]:
            mlp_type_str = mlp_type_str + elem.capitalize()
            if hasattr(sys.modules[__name__], mlp_type_str):
                mlp_type = getattr(sys.modules[__name__], mlp_type_str)
        mlp = mlp_type(self.ipex_config)
        return mlp

    def build_ipex_transformer_config(
        self, config, device, dtype, impl_mode, tp_size, tp_group
    ) -> IPEXTransformerConfig:
        activation_function = "gelu"
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
            intermediate_dim=self.config.hidden_size * 4,
            num_attention_head=self.config.num_attention_heads,
            num_key_value_head=self.config.num_kv_heads,
            max_positions=max(2048, MAX_SEQ_LEN),
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=LlamaRotaryEmbedding,
            rotary_dim=None,
            rotary_half=True,
            rotate_every_two=False,
            use_casual_mask=False,
            activation_function=activation_function,
            ipex_act=ipex_activation,
            norm_eps=self.config.layer_norm_epsilon,
            residual_dropout=None,
            attn_dropout=self.config.attention_dropout,
            enable_bias=False,
            residual_pdrop=None,
            scale_attention=True,
            is_decoder=False,
            do_norm_before=None,
            ln_elementwise_affine=None,
            positional_embedding_base=10000,
            device=self.device,
            dtype=dtype,
            impl=impl_mode,
            tp_size=tp_size,
            tp_group=tp_group,
        )

    def port_attn_parameter(self):
        self.attn.load_parameter(
            qkv_proj=self.module.self_attention.query_key_value,
            out_proj=self.module.self_attention.dense,
        )

    def port_mlp_parameter(self):
        self.mlp.load_parameter(
            self.module.mlp.dense_h_to_4h, self.module.mlp.dense_4h_to_h
        )

    def port_norm_parameter(self):
        self.ln_attn.weight = self.module.ln_attn.weight
        self.ln_attn.bias = self.module.ln_attn.bias
        self.ln_mlp.weight = self.module.ln_mlp.weight
        self.ln_mlp.bias = self.module.ln_mlp.bias

    def transpose_parameter(self):
        self.attn.transpose_parameter()
        self.mlp.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()
        self.attn.cat_qkv()

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        bs = IPEXTransformerAttn.batch_size
        dim = hidden_states.dim()
        if dim == 3:
            beam = hidden_states.shape[0] // bs
            seq = hidden_states.shape[1]
        elif dim == 4:
            beam = hidden_states.shape[1]
            seq = hidden_states.shape[2]
        else:
            print("Unsupported input shape")
            return
        IPEXTransformerAttn.beam_size = beam
        first_token = True if acc_test or layer_past is None else False
        hidden_size = hidden_states.shape[-1]
        hidden_shape = [bs, beam, seq, hidden_size]
        if first_token and beam > 1:
            # for 1st token, keep the original layout
            # reduce the duplicated info in beam dim
            # shape -> [bs*beam, seq, hidden_size]
            # layout -> [bs*beam, seq, hidden_size]
            hidden_states = hidden_states.view(hidden_shape)[:, 0, :, :].contiguous()
            if position_ids is not None:
                position_ids = position_ids.view(bs, beam, position_ids.shape[1])[
                    :, 0, :
                ].view(bs, position_ids.shape[1])
            if attention_mask is not None:
                attention_mask = attention_mask.view(
                    bs,
                    beam,
                    attention_mask.shape[1],
                    attention_mask.shape[2],
                    attention_mask.shape[3],
                )[:, 0, :, :, :].view(
                    bs,
                    attention_mask.shape[1],
                    attention_mask.shape[2],
                    attention_mask.shape[3],
                )
        else:
            # for 2nd to last token, we convert the layout
            # shape -> [bs*beam, seq, hidden_size]
            # convert layout form [bs*beam, seq, hidden_size] to [seq, bs*beam, hidden_size]
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        # revert _convert_to_rw_cache in transformers
        if layer_past is not None:
            batch_size_times_num_heads, kv_length, head_dim = layer_past[0].shape
            num_heads = batch_size_times_num_heads // bs
            layer_past = (
                layer_past[0].view(bs, num_heads, kv_length, head_dim),
                layer_past[1].view(bs, num_heads, kv_length, head_dim),
            )

        residual = hidden_states
        attention_layernorm_out = torch.ops.torch_ipex.fast_layer_norm(
            hidden_states,
            self.ln_attn.normalized_shape,
            self.ln_attn.weight,
            self.ln_attn.bias,
            self.ln_attn.eps,
        )
        mlp_layernorm_out = torch.ops.torch_ipex.fast_layer_norm(
            hidden_states,
            self.ln_mlp.normalized_shape,
            self.ln_mlp.weight,
            self.ln_mlp.bias,
            self.ln_mlp.eps,
        )
        attn_outputs = self.attn(
            hidden_states=attention_layernorm_out,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            residual=residual,
            alibi=alibi,
            first_token=first_token,
            **kwargs,
        )

        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        # residual is already fused into attention
        mlp_output = self.mlp(mlp_layernorm_out, attention_output)
        hidden_states = mlp_output

        if first_token and beam > 1:
            # for 1st token, expand the result with beam
            hidden_states = hidden_states.view(bs, 1, seq, hidden_size)
            hidden_states = hidden_states.expand([bs, beam, seq, hidden_size])
        else:
            # for 2nd to last token, we convert the layout back
            # convert hidden_states form [seq, bs_beam, hidden_size] back to [bs_beam, seq, hidden_size]
            hidden_states = hidden_states.transpose(0, 1)
        if use_cache:
            # revert _convert_cache_to_standard_format in transformers
            layer_past = outputs[0]
            batch_size, num_heads, kv_length, head_dim = layer_past[0].shape
            batch_size_times_num_heads = batch_size * num_heads
            layer_past = (
                layer_past[0].view(batch_size_times_num_heads, kv_length, head_dim),
                layer_past[1].view(batch_size_times_num_heads, kv_length, head_dim),
            )
            outputs = (hidden_states, layer_past) + outputs[1:]
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs
