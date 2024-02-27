import torch
from typing import Optional, Tuple, List
from functools import partial
from .transformer_modules.RoPE import QWenRotaryEmbedding
from .transformer_modules.Norm import QWenRMSNorm


from ._transformers import MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from .transformer_modules.QuantizedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16,
    IPEXTransformerAttnOptimizedInt4,
)  # noqa
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .transformer_modules.GroupedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16Grouped,
)
from .transformer_modules.Decoderblock import IPEXTransformerBlock
from .transformer_modules.Mlp import *  # noqa

# from .transformer_modules.QuantizedMlp import *  # noqa
from .transformer_modules.model_utils import qwen_post_qkv, qwen_sdp

# from intel_extension_for_pytorch.nn.utils._quantize_convert import WeightOnlyLinear
import sys

import os

acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in [
    "1",
    "ON",
    "Y",
    "YES",
    "TRUE",
]


class NewIPEXQWENBlock(IPEXTransformerBlock):
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
        self.attn.post_qkv = partial(qwen_post_qkv, self.attn)
        self.attn.sdp = partial(qwen_sdp, self.attn)
        self.mlp = self.build_mlp_from_config()
        self.input_layernorm = QWenRMSNorm(
            self.ipex_config.embedding_dim, self.ipex_config.norm_eps
        )
        self.post_attn_layernorm = QWenRMSNorm(
            self.ipex_config.embedding_dim, self.ipex_config.norm_eps
        )
        self.port_all_parameters_to_new_module()

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
        for elem in [impl.name, dtype, activation.name, "Qwen"]:
            mlp_type_str = mlp_type_str + elem.capitalize()
            if hasattr(sys.modules[__name__], mlp_type_str):
                mlp_type = getattr(sys.modules[__name__], mlp_type_str)
        return mlp_type(self.ipex_config)

    def build_ipex_transformer_config(
        self, config, device, dtype, impl_mode, tp_size, tp_group
    ) -> IPEXTransformerConfig:
        activation_function = "silu"
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
            max_positions=max(self.config.max_position_embeddings, MAX_SEQ_LEN),
            max_out_positions=MAX_OUT_SEQ_LEN,
            kv_channels=self.config.kv_channels,
            rotary_embedding_class=QWenRotaryEmbedding,
            rotary_emb_base=self.config.rotary_emb_base,
            rotary_pct=self.config.rotary_pct,
            rotary_dim=None,
            rotary_half=True,
            rotate_every_two=False,
            use_casual_mask=False,
            # activation_function=self.config.hidden_act,
            activation_function=activation_function,
            ipex_act=ipex_activation,
            norm_eps=self.config.layer_norm_epsilon,
            residual_dropout=None,
            attn_dropout=self.config.attn_dropout_prob,
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
        embed_dim = self.ipex_config.embedding_dim  # 4096
        dtype = self.ipex_config.dtype
        # if dtype == "int4":
        #     qkv_weight = self.module.attn.c_attn.qweight  # [3*4096, 4096]
        #     qkv_bias = self.module.attn.c_attn.bias
        #     qkv_scale = self.module.attn.c_attn.scales
        #     qkv_zp = None
        #     if hasattr(self.module.attn.c_attn, "qzeros"):
        #         qkv_zp = self.module.attn.c_attn.qzeros
        #     gs = self.module.attn.c_attn.blocksize

        #     q_weight, k_weight, v_weight = qkv_weight.split(int(embed_dim / 2), dim=0)
        #     q_bias, k_bias, v_bias = qkv_bias.view(3, embed_dim).split(1, dim=0)
        #     q_scale, k_scale, v_scale = qkv_scale.split(embed_dim, dim=0)
        #     if qkv_zp is not None:
        #         q_zp, k_zp, v_zp = qkv_zp.split(int(embed_dim / 2), dim=0)

        #     q_proj = IPEXTransformerWOQLinear(q_weight, q_bias, q_scale, q_zp, gs)
        #     k_proj = IPEXTransformerWOQLinear(k_weight, k_bias, k_scale, k_zp, gs)
        #     v_proj = IPEXTransformerWOQLinear(v_weight, v_bias, v_scale, v_zp, gs)
        # else:
        qkv_weight = self.module.attn.c_attn.weight  # [3*4096, 4096]
        qkv_bias = self.module.attn.c_attn.bias
        q_weight, k_weight, v_weight = qkv_weight.split(embed_dim, dim=0)
        q_bias, k_bias, v_bias = qkv_bias.view(3, embed_dim).split(1, dim=0)
        q_proj = IPEXTransformerLinear(q_weight, q_bias)
        k_proj = IPEXTransformerLinear(k_weight, k_bias)
        v_proj = IPEXTransformerLinear(v_weight, v_bias)
        self.attn.load_parameter(q_proj, k_proj, v_proj, self.module.attn.c_proj)

    def port_mlp_parameter(self):
        # IPEXTransformerMLPOptimizedFp16SiluQwen
        self.mlp.load_parameter(
            self.module.mlp.w1,
            self.module.mlp.w2,
            self.module.mlp.c_proj,
        )

    def port_norm_parameter(self):
        self.input_layernorm.weight = self.module.ln_1.weight
        self.post_attn_layernorm.weight = self.module.ln_2.weight

    def transpose_parameter(self):
        self.attn.transpose_parameter()
        self.mlp.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()
        self.attn.cat_qkv()
        # if isinstance(self.module.attn.c_attn, WeightOnlyLinear):
        #     self.module.attn.c_attn.qweight.data = self.attn.qkv_proj_quant.weight.data
        #     self.module.attn.c_attn.bias.data = self.attn.qkv_proj_quant.bias.data
        #     self.module.attn.c_attn.scales.data = self.attn.qkv_proj_quant.scale.data
        #     if hasattr(self.module.attn.c_attn, "qzeros"):
        #         self.module.attn.c_attn.qzeros.data = self.attn.qkv_proj_quant.zp.data
        # else:
        self.module.attn.c_attn.weight.data = self.attn.qkv_proj.weight.data
        self.module.attn.c_attn.bias.data = self.attn.qkv_proj.bias.data

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        rotary_pos_emb_list: Optional[List[List[torch.Tensor]]] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
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
        first_token = True if acc_test or layer_past is None else False
        hidden_size = hidden_states.shape[-1]
        hidden_shape = [bs, beam, seq, hidden_size]
        if first_token and beam > 1:
            # for 1st token, keep the original layout
            # reduce the duplicated info in beam dim
            # shape -> [bs*beam, seq, hidden_size]
            # layout -> [bs*beam, seq, hidden_size]
            hidden_states = hidden_states.view(hidden_shape)[:, 0, :, :].contiguous()
            # if position_ids is not None:
            #     position_ids = position_ids.view(bs, beam, position_ids.shape[1])[
            #         :, 0, :
            #     ].view(bs, position_ids.shape[1])
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
        layernorm_output = self.input_layernorm(hidden_states)

        if layer_past is not None:
            # [bs*beam, seq_len, num_head, head_dim] -> [bs*beam, num_head, seq_len, head_dim]
            layer_past = (layer_past[0].transpose(1, 2), layer_past[1])

        attn_output, present_key_value, self_attn_weights = self.attn(
            hidden_states=layernorm_output,
            layer_past=layer_past,
            rotary_pos_emb_list=rotary_pos_emb_list,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        residual = hidden_states
        layernorm_input = attn_output + residual
        residual = layernorm_input

        layernorm_output = self.post_attn_layernorm(layernorm_input)

        hidden_states = self.mlp(layernorm_output, residual)

        if first_token and beam > 1:
            # for 1st token, expand the result with beam
            hidden_states = hidden_states.view(bs, 1, seq, hidden_size)
            hidden_states = hidden_states.expand([bs, beam, seq, hidden_size])
        else:
            # for 2nd to last token, we convert the layout back
            # convert hidden_states form [seq, beam, hidden_size] back to [beam, seq, hidden_size]
            hidden_states = hidden_states.transpose(0, 1)

        outputs = (hidden_states,)

        if use_cache:
            # present_key_value: (key, value), of shape [bs*beam, num_head, seq_len, head_dim]
            # [bs*beam, num_head, seq_len, head_dim] -> [bs*beam, seq_len, num_head, head_dim]
            present_key_value = (
                present_key_value[0].transpose(1, 2),
                present_key_value[1],
            )
            outputs += (present_key_value,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
