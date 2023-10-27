import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from .transformer_modules.Attention import IPEXTransformerAttnOptimizedFp16, IPEXTransformerAttn

from .transformer_modules.RoPE import PositionalEmbedding

from ._transformers import MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.Mlp import IPEXTransformerBaseMLP, IPEXTransformerMLPOptimizedFp16
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from .transformer_modules.QuantizedAttention import IPEXTransformerAttnNaive, IPEXTransformerAttnOptimizedFp16, IPEXTransformerAttnOptimizedInt4
from .transformer_modules.CrossedAttention import IPEXTransformerAttnOptimizedFp16Crossed
from transformers.modeling_outputs import CausalLMOutputWithPast
from .transformer_modules.Decoderblock import IPEXTransformerBlock
from .transformer_modules.Mlp import *
import sys
from torch.nn import CrossEntropyLoss
import os
acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in ["1", "ON", "Y", "YES", "TRUE"]


class NewIPEXOPTBlock(IPEXTransformerBlock):
    def __init__(self,
                 module,
                 config,
                 dtype = "fp16",
                 device = "xpu",
                 module_name = "",
                 impl_mode = None,
                 tp_size = 1, 
                 tp_group = None):
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(config, device, dtype, impl_mode, tp_size, tp_group)
        self.attn = self.build_attention_from_config()
        self.mlp = self.build_mlp_from_config()
        self.do_layer_norm_before = self.ipex_config.do_norm_before
        self.self_attn_layer_norm = nn.LayerNorm(self.ipex_config.embedding_dim, elementwise_affine=self.ipex_config.ln_elementwise_affine)
        self.final_layer_norm = nn.LayerNorm(self.ipex_config.embedding_dim, elementwise_affine=self.ipex_config.ln_elementwise_affine)
        self.dropout_p = self.ipex_config.residual_pdrop
        self.port_all_parameters_to_new_module()

    def build_attention_from_config(self):
        dtype = self.ipex_config.dtype
        impl = self.ipex_config.impl
        attn_type = IPEXTransformerAttn
        attn_type_str = "IPEXTransformerAttn"
        for elem in [impl.name, dtype, "crossed"]:
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

    
    def build_ipex_transformer_config(self,
                                      config,
                                      device,
                                      dtype,
                                      impl_mode,
                                      tp_size,
                                      tp_group) -> IPEXTransformerConfig:
        activation_function = self.config.activation_function
        ipex_activation = None
        for act in SupportedActivation:
            if activation_function in act.value:
                ipex_activation = act
                break
        assert ipex_activation is not None, "found unrecognized activation function, can not build ipex config from {}".format(activation_function)

        assert dtype in ["fp16", "int4"], "dtype tag {} passed to optimized_transformers is not supported!".format(dtype)

        return IPEXTransformerConfig(
            embedding_dim = self.config.hidden_size,
            intermediate_dim = self.config.ffn_dim,
            num_attention_head = self.config.num_attention_heads,
            max_positions = max(self.config.max_position_embeddings, MAX_SEQ_LEN),
            max_out_positions = MAX_OUT_SEQ_LEN,
            rotary_embedding_class = PositionalEmbedding,
            rotary_dim = None,
            rotary_half=False,
            rotate_every_two=False,
            use_casual_mask = False,
            activation_function = self.config.activation_function,
            ipex_act = ipex_activation,
            norm_eps = None,
            residual_dropout = self.config.dropout,
            attn_dropout = None,
            enable_bias = self.config.enable_bias,
            residual_pdrop = self.config.dropout,
            scale_attention = True,
            is_decoder = True,
            do_norm_before = self.config.do_layer_norm_before,
            ln_elementwise_affine = self.config.layer_norm_elementwise_affine,
            positional_embedding_base = 10000,
            device = self.device,
            dtype = dtype,
            impl = impl_mode,
            tp_size = tp_size,
            tp_group = tp_group
        )

    def port_attn_parameter(self):
        # IPEXTransformerAttnOptimizedFp16Opt IPEXTransformerAttnOptimizedFp16
        self.attn.load_parameter(self.module.self_attn.q_proj, self.module.self_attn.k_proj, self.module.self_attn.v_proj, self.module.self_attn.out_proj)

    def port_mlp_parameter(self):
        # IPEXTransformerMLPOptimizedFp16ReluOpt IPEXTransformerAttnOptimizedFp16
        self.mlp.load_parameter(self.module.fc1, self.module.fc2)

    def port_norm_parameter(self):
        self.self_attn_layer_norm.weight = self.module.self_attn_layer_norm.weight
        self.self_attn_layer_norm.bias = self.module.self_attn_layer_norm.bias
        self.final_layer_norm.weight = self.module.final_layer_norm.weight
        self.final_layer_norm.bias = self.module.final_layer_norm.bias

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
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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
        first_token = True if acc_test or past_key_value is None else False
        hidden_size = hidden_states.shape[-1]
        hidden_shape = [bs, beam, seq, hidden_size]
        if first_token and beam > 1:
            # for 1st token, keep the original layout
            # reduce the duplicated info in beam dim
            # shape -> [bs*beam, seq, hidden_size]
            # layout -> [bs*beam, seq, hidden_size]
            hidden_states = hidden_states.view(hidden_shape)[:, 0, :, :].contiguous()
            if attention_mask is not None:
                shape = attention_mask.shape
                attention_mask = attention_mask.view(bs, beam, shape[1], shape[2], shape[3])[:,0,:,:,:].view(bs, shape[1], shape[2], shape[3])

        else:
            # for 2nd to last token, we convert the layout
            # shape -> [bs*beam, seq, hidden_size]
            # convert layout form [bs*beam, seq, hidden_size] to [seq, bs*beam, hidden_size]
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, present_key_value, self_attn_weights = self.attn(
            hidden_states=hidden_states,
            layer_past=past_key_value,
            attention_mask=attention_mask,
            head_mask=layer_head_mask,
            output_attentions=output_attentions,
            residual=residual,
            first_token=first_token)

        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)

        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

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
            outputs += (present_key_value, )
        return outputs
