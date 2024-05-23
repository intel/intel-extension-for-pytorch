import torch
from typing import Optional, Tuple
from .transformer_modules.RoPE import ChatGLMRotaryEmbedding
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
from .transformer_modules.Attention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16ChatGLM,
)
from .transformer_modules.QuantizedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16,
    IPEXTransformerAttnOptimizedInt4,
)  # noqa
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .transformer_modules.GroupedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16Grouped,
    IPEXTransformerAttnOptimizedFp16GroupedChatGLM,
)
from .transformer_modules.Decoderblock import IPEXTransformerBlock
from .transformer_modules.Mlp import *  # noqa
import sys


# return repeat_interleave of cache compared to original
class NewIPEXRotaryEmbedding(nn.Module):
    def __init__(self, module, config, device="xpu"):
        super().__init__()
        dim = module.dim
        original_impl = module.original_impl

        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, dim, 2, device=device).to(dtype=config.torch_dtype)
                / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(
        self,
        seq_len: int,
        n_elem: int,
        dtype: torch.dtype,
        device: torch.device,
        base: int = 10000,
    ):
        """Enhanced Transformer with Rotary Position Embedding.
        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (
            base
            ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem)
        )

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return torch.repeat_interleave(cache, 2, -2)

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len,
            self.dim,
            dtype=self.inv_freq.dtype,
            device=self.inv_freq.device,
        )


def prepare_inputs_for_generation(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    is_first_forward: bool = True,
    **kwargs,
) -> dict:
    # only last token for input_ids if past is not None
    if position_ids is None:
        position_ids = self.get_position_ids(attention_mask, device=input_ids.device)
    if not is_first_forward:
        if past_key_values is not None:
            position_ids = position_ids[..., -1:]
            input_ids = input_ids[:, -1:]
    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "return_last_logit": True,
        "use_cache": use_cache,
    }


class NewIPEXCHATGLMBlock(IPEXTransformerBlock):
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
        self.mlp = self.build_mlp_from_config()

        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
        )
        self.post_attn_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
        )

        self.port_all_parameters_to_new_module()

    def build_attention_from_config(self):
        dtype = self.ipex_config.dtype
        impl = self.ipex_config.impl
        attn_type = IPEXTransformerAttn
        attn_type_str = "IPEXTransformerAttn"
        attn_list = [impl.name, dtype]
        if self.ipex_config.multi_query_attention:
            attn_list.append("Grouped")
        attn_list.append("ChatGLM")

        for elem in attn_list:
            attn_type_str = attn_type_str + elem.capitalize()[0] + elem[1:]
            if hasattr(sys.modules[__name__], attn_type_str):
                attn_type = getattr(sys.modules[__name__], attn_type_str)
        return attn_type(self.ipex_config)

    def build_mlp_from_config(self):
        dtype = self.ipex_config.dtype
        impl = self.ipex_config.impl
        activation = self.ipex_config.ipex_act
        mlp_type = IPEXTransformerMLP
        mlp_type_str = "IPEXTransformerMLP"
        for elem in [impl.name, dtype, activation.name, "ChatGLM"]:
            mlp_type_str = mlp_type_str + elem.capitalize()[0] + elem[1:]
            if hasattr(sys.modules[__name__], mlp_type_str):
                mlp_type = getattr(sys.modules[__name__], mlp_type_str)
        return mlp_type(self.ipex_config)

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
            apply_residual_connection_post_layernorm=self.config.apply_residual_connection_post_layernorm,
            embedding_dim=self.config.hidden_size,
            intermediate_dim=self.config.ffn_hidden_size,
            num_key_value_head=self.config.multi_query_group_num,
            norm_eps=self.config.layernorm_epsilon,
            multi_query_attention=self.config.multi_query_attention,
            num_attention_head=self.config.num_attention_heads,
            rmsnorm=self.config.rmsnorm,
            # transformers==4.31.0
            max_positions=MAX_SEQ_LEN,
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=ChatGLMRotaryEmbedding,
            rotary_dim=None,
            rotary_half=True,
            rotate_every_two=False,
            use_casual_mask=False,
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
            qkv_proj=self.module.self_attention.query_key_value,
            out_proj=self.module.self_attention.dense,
        )

    def port_mlp_parameter(self):
        # IPEXTransformerMLPOptimizedFp16SiluChatGLM
        self.mlp.load_parameter(
            self.module.mlp.dense_h_to_4h,
            self.module.mlp.dense_4h_to_h,
        )

    def port_norm_parameter(self):
        self.input_layernorm.weight = self.module.input_layernorm.weight
        self.post_attn_layernorm.weight = self.module.post_attention_layernorm.weight

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
        rotary_pos_emb: Optional[torch.LongTensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        # hidden_states:  [seq, bs*beam, hidden_size]
        # rotary_pos_emb:   [seq, bs*beam, seq, 2]
        # attention_mask: [bs*beam, head, q_seq, kv_seq]
        bs = IPEXTransformerAttn.batch_size
        dim = hidden_states.dim()
        if dim == 3:
            beam = hidden_states.shape[1] // bs
            seq = hidden_states.shape[0]
        elif dim == 4:
            beam = hidden_states.shape[2]
            seq = hidden_states.shape[0]
        else:
            print("Unsupported input shape")
            return

        IPEXTransformerAttn.beam_size = beam
        first_token = True if kv_cache is None else False

        hidden_size = hidden_states.shape[-1]
        hidden_shape = [bs, beam, seq, hidden_size]
        if first_token and beam > 1:
            # for 1st token, keep the original layout
            # reduce the duplicated info in beam dim
            # shape -> [bs*beam, seq, hidden_size]
            # layout -> [bs*beam, seq, hidden_size]
            hidden_states = (
                hidden_states.transpose(0, 1)
                .view(hidden_shape)[:, 0, :, :]
                .contiguous()
            )
            if rotary_pos_emb is not None:
                rotary_pos_emb = (
                    rotary_pos_emb.transpose(0, 1)
                    .view(
                        bs,
                        beam,
                        rotary_pos_emb.shape[0],
                        rotary_pos_emb.shape[2],
                        rotary_pos_emb.shape[3],
                    )[:, 0, :, :, :]
                    .view(
                        bs,
                        rotary_pos_emb.shape[0],
                        rotary_pos_emb.shape[2],
                        rotary_pos_emb.shape[3],
                    )
                )
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

        if rotary_pos_emb is not None:
            rotary_pos_emb = rotary_pos_emb.unsqueeze(2)

        layernorm_output = self.input_layernorm(hidden_states)

        if self.ipex_config.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input, present_key_value, self_attn_weights = self.attn(
            hidden_states=layernorm_output,
            attention_mask=attention_mask,
            position_ids=rotary_pos_emb,
            layer_past=kv_cache,
            use_cache=use_cache,
            residual=residual,
            first_token=first_token,
        )

        layernorm_output = self.post_attn_layernorm(layernorm_input)

        if self.config.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        mlp_output = self.mlp(layernorm_output, residual)

        if first_token and beam > 1:
            # for 1st token, expand the result with beam
            mlp_output = (
                mlp_output.view(bs, 1, seq, hidden_size)
                .expand([bs, beam, seq, hidden_size])
                .view(bs * beam, seq, hidden_size)
                .squeeze()
                .transpose(0, 1)
            )
        outputs = (mlp_output,)

        if use_cache:
            outputs += (present_key_value,)
        return outputs
