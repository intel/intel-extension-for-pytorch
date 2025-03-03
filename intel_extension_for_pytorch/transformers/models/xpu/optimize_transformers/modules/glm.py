import torch
from typing import Optional, Tuple

from .transformer_modules.RoPE import GLMRotaryEmbedding
from .transformer_modules.Norm import LlamaRMSNorm

from ._transformers import MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.Mlp import (  # noqa F401
    IPEXTransformerBaseMLP,
    IPEXTransformerMLPOptimizedFp16,
)
from ._transformer_configuration import (
    IPEXTransformerConfigChatGLM,
    SupportedActivation,
)
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .transformer_modules.GroupedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16Grouped,
)
from .transformer_modules.DecoderBlock import IPEXTransformerBlock
from .transformer_modules.Mlp import *  # noqa
from .transformer_modules.model_utils import (
    xpu_gemm_use_xetla,
)

from .transformer_modules.XPUAttentionfp16 import (
    IPEXAttention,
    IPEXGroupedAttention,
)
from .transformer_modules.XPUAttentionInt4 import (
    IPEXAttentionInt4,
    IPEXAttentionInt4OneDNN,
)


# return repeat_interleave of cache compared to original
class NewIPEXGlmRotaryEmbedding(nn.Module):
    def __init__(self, module, config, device="xpu"):
        super().__init__()

    def forward(self, max_seq_len, offset=0):
        return None


class NewIPEXGlmBlock(IPEXTransformerBlock):
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
        grouped = True if self.ipex_config.multi_query_attention else False

        if dtype == "fp16":
            self.attn = (
                IPEXGroupedAttention(self.ipex_config, module.self_attn.layer_idx)
                if grouped
                else IPEXAttention(self.ipex_config, module.self_attn.layer_idx)
            )
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

        self.mlp = self.build_mlp_from_config("ChatGLM")

        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attn_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.port_all_parameters_to_new_module()
        self.layer_idx = kwargs.get("layer_idx", None)

    def build_ipex_transformer_config(
        self, config, device, dtype, impl_mode, tp_size, tp_group
    ) -> IPEXTransformerConfigChatGLM:
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

        return IPEXTransformerConfigChatGLM(
            embedding_dim=self.config.hidden_size,
            intermediate_dim=self.config.intermediate_size,
            num_key_value_head=self.config.num_key_value_heads,
            norm_eps=self.config.rms_norm_eps,
            multi_query_attention=self.config.num_key_value_heads
            != self.config.num_attention_heads,
            num_attention_head=self.config.num_attention_heads,
            max_positions=MAX_SEQ_LEN,
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=GLMRotaryEmbedding,
            rotary_dim=None,
            rotary_half=True,
            rotate_every_two=False,
            use_causal_mask=False,
            activation_function="silu",
            ipex_act=ipex_activation,
            residual_dropout=None,
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
            self.module.self_attn.q_proj,
            self.module.self_attn.k_proj,
            self.module.self_attn.v_proj,
            self.module.self_attn.o_proj,
        )

    def port_mlp_parameter(self):
        # IPEXTransformerMLPOptimizedFp16SiluChatGLM
        self.mlp.load_parameter(
            self.module.mlp.gate_up_proj,
            self.module.mlp.down_proj,
        )

    def port_norm_parameter(self):
        self.input_layernorm.weight = self.module.input_layernorm.weight
        self.post_attn_layernorm.weight = self.module.post_attention_layernorm.weight

    def transpose_parameter(self):
        dtype = self.ipex_config.dtype
        if self.ipex_config.multi_query_attention and dtype == "fp16":
            self.attn.transpose_parameter()
        else:
            self.attn.transpose_parameter(dtype=dtype)

        self.mlp.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        dtype = self.ipex_config.dtype
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()
        self.attn.cat_qkv()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        # hidden_states:  [bs*beam, seq, hidden_size]
        # rotary_pos_emb:   [bs*beam, seq, seq, 2]
        # attention_mask: [bs*beam, head, q_seq, kv_seq]
        bs = IPEXTransformerAttn.batch_size

        b, seq, hidden_size = hidden_states.shape

        IPEXTransformerAttn.beam_size = b // bs
        beam = IPEXTransformerAttn.beam_size
        first_token = True if seq > 1 else False
        if first_token and beam > 1:
            hidden_states = hidden_states.view(bs, beam, seq, hidden_size)[
                :, 0, :, :
            ].contiguous()
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

        if (
            hasattr(past_key_value, "max_batch_size")
            and past_key_value.max_batch_size < b
        ):
            repeat_cnt = b // past_key_value.max_batch_size
            for i in range(len(past_key_value.key_cache)):
                past_key_value.key_cache[i] = past_key_value.key_cache[i].repeat(
                    1, repeat_cnt, 1, 1
                )
                past_key_value.value_cache[i] = past_key_value.value_cache[i].repeat(
                    1, repeat_cnt, 1, 1
                )
            past_key_value.max_batch_size = b

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, self_attn_weights = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            residual=residual,
            cache_position=cache_position,
        )

        residual = hidden_states

        hidden_states = self.post_attn_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states, residual)

        if first_token and beam > 1:
            hidden_states = (
                hidden_states.view(bs, 1, seq, hidden_size)
                .expand([bs, beam, seq, hidden_size])
                .view(bs * beam, seq, hidden_size)
                .squeeze()
            )

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs
