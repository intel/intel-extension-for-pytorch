import math
import warnings
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
import sys
import os 
from ._transformers import MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from .transformer_modules.RoPE import PositionalEmbedding
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.Attention import IPEXTransformerAttnNaive, IPEXTransformerAttnOptimizedFp16
from .transformer_modules.QuantizedAttention import IPEXTransformerAttnOptimizedInt4
from .transformer_modules.Mlp import *
from .transformer_modules.Decoderblock import IPEXTransformerBlock
from .transformer_modules.Linear import IPEXTransformerLinear, IPEXTransformerQLinear

import os
acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in ["1", "ON", "Y", "YES", "TRUE"]


class NewIPEXBloomBlock(IPEXTransformerBlock):
    def __init__(self,
                 module,
                 config: IPEXTransformerConfig,
                 dtype="fp16",
                 device="xpu",
                 module_name="",
                 impl_mode = None,
                 tp_size=1,
                 tp_group=None):
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(config, device, dtype, impl_mode, tp_size, tp_group)
        self.attn = self.build_attention_from_config()
        self.mlp = self.build_mlp_from_config()
        self.input_layernorm = nn.LayerNorm(self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps)
        self.port_all_parameters_to_new_module()

    def build_ipex_transformer_config(self,
                                      config,
                                      device,
                                      dtype,
                                      impl_mode,
                                      tp_size,
                                      tp_group):
        activation_function = "bloom_gelu"
        ipex_activation = None
        for act in SupportedActivation:
            if activation_function in act.value:
                ipex_activation = act
                break
        assert ipex_activation is not None, "found unrecognized activation function, can not build ipex config from {}".format(activation_function)
        assert dtype in ["fp16", "int4"], "dtype tag {} passed to optimized_transformers is not supported!".format(dtype)

        return IPEXTransformerConfig(
            embedding_dim = self.config.hidden_size,
            intermediate_dim = 4 * self.config.hidden_size,
            num_attention_head = self.config.n_head,
            max_positions = max(2048, MAX_SEQ_LEN),
            max_out_positions = MAX_OUT_SEQ_LEN,
            rotary_embedding_class = PositionalEmbedding,
            rotary_dim = None,
            use_casual_mask = True,
            activation_function = activation_function,
            ipex_act = ipex_activation,
            norm_eps = self.config.layer_norm_epsilon,
            residual_dropout = None,
            attn_dropout = self.config.attention_dropout,
            enable_bias = False,
            residual_pdrop = None,
            scale_attention = True,
            is_decoder = True,
            do_norm_before = self.config.apply_residual_connection_post_layernorm,
            ln_elementwise_affine = None,
            dtype = dtype,
            impl = impl_mode,
            tp_size = tp_size,
            tp_group = tp_group
        )

    def build_attention_from_config(self):
        dtype = self.ipex_config.dtype
        impl = self.ipex_config.impl
        attn_type = IPEXTransformerAttn
        attn_type_str = "IPEXTransformerAttn"
        for elem in [impl.name, dtype]:
            attn_type_str = attn_type_str + elem.capitalize()
            if hasattr(sys.modules[__name__], attn_type_str):
                attn_type = getattr(sys.modules[__name__], attn_type_str)
        return attn_type(self.ipex_config)

    def build_mlp_from_config(self):
        dtype = self.ipex_config.dtype
        impl = self.ipex_config.impl
        activation = self.ipex_config.ipex_act
        mlp_type = IPEXTransformerMLP
        mlp_type_str = "IPEXTransformerMLP"
        for elem in [impl.name, dtype, activation.name]:
            mlp_type_str = mlp_type_str + elem.capitalize()
            if hasattr(sys.modules[__name__], mlp_type_str):
                mlp_type = getattr(sys.modules[__name__], mlp_type_str)
        return mlp_type(self.ipex_config)

    def port_attn_parameter(self):
        embed_dim = self.ipex_config.embedding_dim
        num_head = self.ipex_config.num_attention_head // self.ipex_config.tp_size
        weight_shape = [num_head, 3, -1, embed_dim]
        bias_shape = [num_head, 3, -1]
        qkv_weight = self.module.self_attention.query_key_value.weight
        qkv_bias = self.module.self_attention.query_key_value.bias
        qkv_weight.data = qkv_weight.view(weight_shape).transpose(0, 1).contiguous().view([3, -1, embed_dim]).contiguous()
        qkv_bias.data = qkv_bias.view(bias_shape).transpose(0, 1).contiguous().view([3, -1]).contiguous()
        q_proj = IPEXTransformerLinear(qkv_weight[0], qkv_bias[0])
        k_proj = IPEXTransformerLinear(qkv_weight[1], qkv_bias[1])
        v_proj = IPEXTransformerLinear(qkv_weight[2], qkv_bias[2])
        self.attn.load_parameter(q_proj, k_proj, v_proj, self.module.self_attention.dense)

    def port_mlp_parameter(self):
        self.mlp.load_parameter(self.module.mlp.dense_h_to_4h, self.module.mlp.dense_4h_to_h)
    
    def port_norm_parameter(self):
        self.input_layernorm.weight = self.module.input_layernorm.weight
        self.input_layernorm.bias   = self.module.input_layernorm.bias
        self.post_attention_layernorm.weight = self.module.post_attention_layernorm.weight
        self.post_attention_layernorm.bias   = self.module.post_attention_layernorm.bias
    
    def transpose_parameter(self):
        self.attn.transpose_parameter()
        self.mlp.transpose_parameter()
    
    def port_all_parameters_to_new_module(self):
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()
        self.attn.cat_qkv()
        self.module.self_attention.query_key_value.weight.data = self.attn.qkv_proj.weight.data
        self.module.self_attention.query_key_value.bias.data = self.attn.qkv_proj.bias.data


    def release_resources(self):
        self.attn.release_resources()
        self.mlp.release_resources()
    
    def print_rank_x(self, str):
        if dist.get_rank() == 1:
            print(str)

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
            if attention_mask is not None:
                attention_mask = attention_mask.view(bs, beam, attention_mask.shape[1], attention_mask.shape[2], attention_mask.shape[3])[:,0,:,:,:].view(bs, attention_mask.shape[1], attention_mask.shape[2], attention_mask.shape[3])
        else:
            # for 2nd to last token, we convert the layout
            # shape -> [bs*beam, seq, hidden_size]
            # convert layout form [bs*beam, seq, hidden_size] to [seq, bs*beam, hidden_size]
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        layernorm_output = torch.ops.torch_ipex.fast_layer_norm(hidden_states, self.input_layernorm.normalized_shape, self.input_layernorm.weight, self.input_layernorm.bias, self.input_layernorm.eps)
        if self.ipex_config.do_norm_before:
            residual = layernorm_output
        else:
            residual = hidden_states
        attn_outputs = self.attn(
            hidden_states = layernorm_output,
            layer_past = layer_past,
            attention_mask = attention_mask,
            head_mask = head_mask,
            use_cache = use_cache,
            output_attentions = output_attentions,
            residual=residual,
            alibi=alibi,
            first_token=first_token
        )

        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        layernorm_output = torch.ops.torch_ipex.fast_layer_norm(attention_output, self.post_attention_layernorm.normalized_shape, self.post_attention_layernorm.weight, self.post_attention_layernorm.bias, self.post_attention_layernorm.eps)
        if self.ipex_config.do_norm_before:
            residual = layernorm_output
        else:
            residual = attention_output

        hidden_states = self.mlp(layernorm_output, residual)

        if first_token and beam > 1:
            # for 1st token, expand the result with beam
            hidden_states = hidden_states.view(bs, 1, seq, hidden_size)
            hidden_states = hidden_states.expand([bs, beam, seq, hidden_size])
        else:
            # for 2nd to last token, we convert the layout back
            # convert hidden_states form [seq, beam, hidden_size] back to [beam, seq, hidden_size]
            hidden_states = hidden_states.transpose(0, 1)
        if use_cache:
            outputs = (hidden_states, ) + outputs
        else:
            outputs = (hidden_states, ) + outputs[1:]
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

