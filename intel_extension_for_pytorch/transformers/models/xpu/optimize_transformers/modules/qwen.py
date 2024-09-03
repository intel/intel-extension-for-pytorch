import torch
from typing import Optional, Tuple, List
from functools import partial
from .transformer_modules.RoPE import Qwen2RotaryEmbedding
from .transformer_modules.Norm import QWenRMSNorm


from ._transformers import MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.Mlp import (  # noqa F401
    IPEXTransformerBaseMLP,
    IPEXTransformerMLPOptimizedFp16,
)
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from .transformer_modules.QuantizedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16,
    IPEXTransformerAttnOptimizedInt4,
)  # noqa
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .transformer_modules.GroupedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16Grouped,
)
from .transformer_modules.DecoderBlock import IPEXTransformerBlock
from .transformer_modules.Mlp import *  # noqa
from .transformer_modules.QuantizedMlp import *  # noqa
from .transformer_modules.model_utils import (
    qwen_post_qkv,
    qwen_sdp,
    load_attn_fused_qkv_params,
    transpose_attn_fused_qkv_params,
)


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
        **kwargs,
    ):
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(
            config, device, dtype, impl_mode, tp_size, tp_group
        )
        self.attn = self.build_attention_from_config()
        self.attn.post_qkv = partial(qwen_post_qkv, self.attn)
        self.attn.sdp = partial(qwen_sdp, self.attn)
        self.mlp = self.build_mlp_from_config("Qwen")
        self.input_layernorm = QWenRMSNorm(
            self.ipex_config.embedding_dim, self.ipex_config.norm_eps
        )
        self.post_attn_layernorm = QWenRMSNorm(
            self.ipex_config.embedding_dim, self.ipex_config.norm_eps
        )
        self.port_all_parameters_to_new_module()

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
            rotary_embedding_class=Qwen2RotaryEmbedding,
            rotary_pct=self.config.rotary_pct,
            rotary_dim=None,
            rotary_half=True,
            rotate_every_two=False,
            use_causal_mask=False,
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
        self.attn.load_parameter(
            self.module.attn.c_attn,
            self.module.attn.c_proj,
            dtype=self.ipex_config.dtype,
        )

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
        first_token = True if layer_past is None else False
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
