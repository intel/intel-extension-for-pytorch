from typing import Optional, Tuple
from functools import partial
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.activations import get_activation
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from ._transformers import MAX_OUT_SEQ_LEN, MAX_SEQ_LEN
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.DecoderBlock import IPEXTransformerBlock
from .transformer_modules.GroupedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16Grouped,
)
from .transformer_modules.Mlp import *  # noqa
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .transformer_modules.QuantizedAttention import (  # noqa F401; noqa
    IPEXTransformerAttnOptimizedFp16,
)
from .transformer_modules.RoPE import LlamaRotaryEmbedding
from .transformer_modules.XPUAttentionfp16 import (
    IPEXAttention,
    IPEXGroupedAttention,
)

from .transformer_modules.model_utils import (
    load_attn_fused_qkv_params,
    chatglm_load_attn_params_grouped,
    transpose_attn_fused_qkv_params,
)


def dropout_add(
    x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool
) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`):
            input tensor
        residual (`torch.tensor`):
            residual tensor
        prob (`float`):
            dropout probability
        training (`bool`):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class FalconLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_states = input @ self.weight.T
        if self.bias is None:
            return hidden_states
        return hidden_states + self.bias


class FalconMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size

        self.dense_h_to_4h = FalconLinear(
            hidden_size, config.ffn_hidden_size, bias=config.bias
        )
        self.act = get_activation(config.activation)
        self.dense_4h_to_h = FalconLinear(
            config.ffn_hidden_size, hidden_size, bias=config.bias
        )
        self.hidden_dropout = config.hidden_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x


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
        **kwargs,
    ):
        self.new_decoder_architecture = (
            True if config.new_decoder_architecture else False
        )
        self.num_ln_in_parallel_attn = (
            2
            if config.num_ln_in_parallel_attn is None and self.new_decoder_architecture
            else config.num_ln_in_parallel_attn
        )
        self.parallel_attn = config.parallel_attn
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(
            config, device, dtype, impl_mode, tp_size, tp_group
        )

        self.grouped = False
        if dtype == "fp16":
            if (
                self.ipex_config.num_attention_head
                > self.ipex_config.num_key_value_head
            ):
                self.grouped = True
                self.attn = IPEXGroupedAttention(
                    self.ipex_config, module.self_attention.layer_idx
                )
            else:
                self.attn = IPEXAttention(
                    self.ipex_config, module.self_attention.layer_idx
                )
        else:
            raise NotImplementedError(
                "IPEXAttention dose not support this modelType {} !".format(dtype)
            )

        if not self.new_decoder_architecture:
            if self.grouped:
                self.attn.load_parameter = partial(
                    chatglm_load_attn_params_grouped, self.attn
                )
            else:
                self.attn.load_parameter = partial(
                    load_attn_fused_qkv_params, self.attn
                )
                self.attn.transpose_parameter = partial(
                    transpose_attn_fused_qkv_params, self.attn
                )

        self.mlp = (
            FalconMLP(config)
            if not self.new_decoder_architecture
            else self.build_mlp_from_config("Falcon")
        )

        if not self.parallel_attn:
            self.post_attention_layernorm = nn.LayerNorm(
                self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
            )
            self.input_layernorm = nn.LayerNorm(
                self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
            )
        else:
            if self.num_ln_in_parallel_attn == 2:
                self.ln_attn = nn.LayerNorm(
                    self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
                )
                self.ln_mlp = nn.LayerNorm(
                    self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
                )
            else:
                self.input_layernorm = nn.LayerNorm(
                    self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
                )

        self.port_all_parameters_to_new_module()
        self.layer_idx = kwargs.get("layer_idx", None)

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
            num_key_value_head=self.config.num_key_value_heads,
            max_positions=max(2048, MAX_SEQ_LEN),
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=LlamaRotaryEmbedding,
            rotary_dim=None,
            rotary_half=True,
            rotate_every_two=False,
            use_causal_mask=False,
            activation_function=activation_function,
            ipex_act=ipex_activation,
            norm_eps=self.config.layer_norm_epsilon,
            residual_dropout=self.config.hidden_dropout,
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
        if self.new_decoder_architecture:
            self.attn.load_parameter(
                qkv_proj=self.module.self_attention.query_key_value,
                out_proj=self.module.self_attention.dense,
            )
        else:
            self.attn.load_parameter(
                self.module.self_attention.query_key_value,
                self.module.self_attention.dense,
                dtype=self.ipex_config.dtype,
            )

    def port_mlp_parameter(self):
        if self.new_decoder_architecture:
            self.mlp.load_parameter(
                self.module.mlp.dense_h_to_4h, self.module.mlp.dense_4h_to_h
            )
        else:
            self.mlp.dense_h_to_4h = self.module.mlp.dense_h_to_4h
            self.mlp.dense_4h_to_h = self.module.mlp.dense_4h_to_h

    def port_norm_parameter(self):
        if not self.parallel_attn:
            self.post_attention_layernorm.weight = (
                self.module.post_attention_layernorm.weight
            )
            self.post_attention_layernorm.bias = (
                self.module.post_attention_layernorm.bias
            )
            self.input_layernorm.weight = self.module.input_layernorm.weight
            self.input_layernorm.bias = self.module.input_layernorm.bias
        else:
            if self.num_ln_in_parallel_attn == 2:
                self.ln_attn.weight = self.module.ln_attn.weight
                self.ln_attn.bias = self.module.ln_attn.bias
                self.ln_mlp.weight = self.module.ln_mlp.weight
                self.ln_mlp.bias = self.module.ln_mlp.bias
            else:
                self.input_layernorm.weight = self.module.input_layernorm.weight
                self.input_layernorm.bias = self.module.input_layernorm.bias

    def transpose_parameter(self):
        if self.new_decoder_architecture:
            self.attn.transpose_parameter()
            self.mlp.transpose_parameter()
        else:
            if not self.grouped:
                dtype = self.ipex_config.dtype
                self.attn.transpose_parameter(dtype=dtype)
            else:
                self.attn.transpose_parameter()

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
        cache_position: Optional[torch.LongTensor] = None,
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
        IPEXTransformerAttn.beam_size = hidden_states.shape[0] // bs

        residual = hidden_states
        if self.new_decoder_architecture and self.num_ln_in_parallel_attn == 2:
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
        else:
            attention_layernorm_out = torch.ops.torch_ipex.fast_layer_norm(
                hidden_states,
                self.input_layernorm.normalized_shape,
                self.input_layernorm.weight,
                self.input_layernorm.bias,
                self.input_layernorm.eps,
            )
        attn_outputs = self.attn(
            hidden_states=attention_layernorm_out,
            past_key_value=layer_past,
            attention_mask=None,
            cache_position=cache_position,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            alibi=alibi,
            **kwargs,
        )

        attention_output = attn_outputs[0]
        if not self.new_decoder_architecture:
            if self.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(
                    attention_output,
                    residual,
                    self.ipex_config.attn_dropout,
                    training=self.training,
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        if (
            self.new_decoder_architecture
            and self.parallel_attn
            and self.num_ln_in_parallel_attn == 1
        ):
            mlp_layernorm_out = attention_layernorm_out
        outputs = attn_outputs[1:]

        # residual is already fused into attention
        mlp_output = self.mlp(mlp_layernorm_out)
        if self.new_decoder_architecture or self.parallel_attn:
            mlp_output += attention_output
        output = dropout_add(
            mlp_output,
            residual,
            self.ipex_config.residual_dropout,
            training=self.training,
        )
        next_cache = None
        if use_cache:
            layer_past = outputs[0]
            outputs = (output, layer_past) + outputs[1:]
        else:
            outputs = (output,) + outputs[1:]
        return outputs
