import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from functools import partial

from .transformer_modules.RoPE import Phi3RotaryEmbedding
from .transformer_modules.Norm import LlamaRMSNorm

Phi3RMSNorm = LlamaRMSNorm

from ._transformers import MAX_OUT_SEQ_LEN
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
    qwen_load_attn_params_fp16,
    qwen_transpose_attn_params_fp16,
    qwen_load_attn_params_int4,
    qwen_transpose_attn_params_int4,
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
        grouped = False
        if self.ipex_config.num_attention_head > self.ipex_config.num_key_value_head:
            grouped = True
        self.attn = self.build_attention_from_config(grouped=grouped)
        # self.attn.post_qkv = partial(phi3_post_qkv, self.attn)
        self.attn.position_embed = self.ipex_config.rotary_embedding_class(
            self.ipex_config, torch.float16
        )
        self.mlp = self.build_mlp_from_config("phi3")
        self.resid_attn_dropout = nn.Dropout(self.ipex_config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(self.ipex_config.resid_pdrop)
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
            use_casual_mask=False,
            activation_function=activation_function,
            ipex_act=ipex_activation,
            norm_eps=self.config.rms_norm_eps,
            resid_pdrop=self.config.resid_pdrop,
            attn_dropout=self.config.attention_dropout,
            scale_attention=True,
            dtype=dtype,
            impl=impl_mode,
            tp_size=tp_size,
            tp_group=tp_group,
        )

    def port_attn_parameter(self):
        self.attn.load_parameter(
            self.module.self_attn.qkv_proj, self.module.self_attn.o_proj
        )

    def port_mlp_parameter(self):
        self.mlp.load_parameter(self.module.mlp.gate_up_proj, self.module.mlp.down_proj)

    def port_norm_parameter(self):
        self.input_layernorm.weight = self.module.input_layernorm.weight
        # self.input_layernorm.bias = self.module.input_layernorm.bias
        self.post_attention_layernorm.weight = (
            self.module.post_attention_layernorm.weight
        )
        # self.post_attention_layernorm.bias = self.module.post_attention_layernorm.bias

    def port_dropout_parameter(self):
        self.resid_attn_dropout.p = self.module.resid_attn_dropout.p
        self.resid_attn_dropout.inplace = self.module.resid_attn_dropout.inplace
        self.resid_mlp_dropout.p = self.module.resid_mlp_dropout.p
        self.resid_mlp_dropout.inplace = self.module.resid_mlp_dropout.inplace

    def transpose_parameter(self):
        self.attn.transpose_parameter()
        self.mlp.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        dtype = self.ipex_config.dtype
        if dtype == "fp16":
            self.attn.load_parameter = partial(qwen_load_attn_params_fp16, self.attn)
            self.attn.transpose_parameter = partial(
                qwen_transpose_attn_params_fp16, self.attn
            )
        elif dtype == "int4":
            self.attn.load_parameter = partial(qwen_load_attn_params_int4, self.attn)
            self.attn.transpose_parameter = partial(
                qwen_transpose_attn_params_int4, self.attn
            )
        super().port_all_parameters_to_new_module()
        self.port_dropout_parameter()
        if self.ipex_config.transpose:
            self.transpose_parameter()
        # self.attn.cat_qkv()

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        # print("hidden_states.shape: ", hidden_states.shape)
        # hidden_states:  [bs*beam, seq, hidden_size]
        # position_ids:   [bs*beam, seq]
        # attention_mask: [bs*beam, head, q_seq, kv_seq]
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
        IPEXTransformerMLP.beam_size = beam
        first_token = True if seq > 1 else False
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

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_idx < len(past_key_value):
            layer_past = (
                past_key_value.key_cache[self.layer_idx],
                past_key_value.value_cache[self.layer_idx],
            )
        else:
            layer_past = None

        hidden_states, present_key_value, self_attn_weights = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        if self.layer_idx >= len(past_key_value.key_cache):
            past_key_value.update(
                present_key_value[0], present_key_value[1], self.layer_idx
            )
        else:
            past_key_value.update(
                present_key_value[0][:, :, -1, :].unsqueeze(-2),
                present_key_value[1][:, :, -1, :].unsqueeze(-2),
                self.layer_idx,
            )

        hidden_states = residual + self.resid_attn_dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        if first_token and beam > 1:
            # for 1st token, expand the result with beam
            hidden_states = hidden_states.view(bs, 1, seq, hidden_size)
            hidden_states = hidden_states.expand([bs, beam, seq, hidden_size])
        else:
            # for 2nd to last token, we convert the layout back
            # convert hidden_states form [seq, beam, hidden_size] back to [beam, seq, hidden_size]
            hidden_states = hidden_states.transpose(0, 1)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs
