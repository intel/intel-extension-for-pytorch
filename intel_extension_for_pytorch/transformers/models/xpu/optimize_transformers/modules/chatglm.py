import torch
from typing import Optional, Tuple
from functools import partial
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

from .transformer_modules.RoPE import ChatGLMRotaryEmbedding
from .transformer_modules.Norm import LlamaRMSNorm
from .transformer_modules.CacheUtils import (
    IPEXStaticCache,
    CacheFormat,
)

from ._transformers import MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.Attention import IPEXTransformerAttnOptimizedFp16
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
    chatglm_load_attn_params_grouped,
    load_attn_fused_qkv_params,
    transpose_attn_fused_qkv_params,
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


def GLMModel_forward(
    self,
    input_ids,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.BoolTensor] = None,
    full_attention_mask: Optional[torch.BoolTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    batch_size, seq_length = input_ids.shape

    if past_key_values is None:
        past_length = 0
    elif isinstance(past_key_values, Cache):
        past_length = past_key_values.get_seq_length()
    else:
        past_length = past_key_values[0][0].shape[2]

    cache_position = torch.arange(
        past_length, past_length + seq_length, device=input_ids.device
    )

    if inputs_embeds is None:
        inputs_embeds = self.embedding(input_ids)

    if full_attention_mask is None:
        if (attention_mask is not None and not attention_mask.all()) or (
            past_length != 0 and seq_length != 1
        ):
            full_attention_mask = self.get_masks(
                input_ids, past_key_values, padding_mask=attention_mask
            )

    # Rotary positional embeddings
    rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
    if position_ids is not None:
        rotary_pos_emb = rotary_pos_emb[position_ids]
    else:
        rotary_pos_emb = rotary_pos_emb[None, :seq_length]

    # Run encoder.
    hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
        inputs_embeds,
        full_attention_mask,
        rotary_pos_emb=rotary_pos_emb,
        kv_caches=past_key_values,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
    )
    if presents is not None and type(presents) is torch.Tensor:
        presents = presents.split(1, dim=0)
        presents = list(presents)
        presents = [list(x.squeeze(0).split(1, dim=0)) for x in presents]
        presents = [tuple([x.squeeze(0) for x in y]) for y in presents]
        presents = tuple(presents)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
            if v is not None
        )

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def GLMTransformer_forward(
    self,
    hidden_states,
    attention_mask,
    rotary_pos_emb,
    kv_caches=None,
    use_cache: Optional[bool] = True,
    output_hidden_states: Optional[bool] = False,
    cache_position: Optional[torch.Tensor] = None,
):
    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    all_self_attentions = None
    all_hidden_states = () if output_hidden_states else None
    for index in range(self.num_layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer = self._get_layer(index)
        if self.gradient_checkpointing and self.training:
            layer_ret = torch.utils.checkpoint.checkpoint(
                layer,
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_caches,
                use_cache,
                use_reentrant=False,
            )
        else:
            layer_ret = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_cache=kv_caches,
                use_cache=use_cache,
                cache_position=cache_position,
            )
        hidden_states, kv_cache = layer_ret
        if use_cache:
            presents = kv_cache

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    # Final layer norm.
    if self.post_layer_norm:
        hidden_states = self.final_layernorm(hidden_states)

    return hidden_states, presents, all_hidden_states, all_self_attentions


# return repeat_interleave of cache compared to original
class NewIPEXChatGLMRotaryEmbedding(nn.Module):
    def __init__(self, module, config, device="xpu"):
        super().__init__()
        dim = module.dim
        original_impl = module.original_impl

        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, dim, 2, device=device).to(dtype=torch.float16) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl
        self.rope_ratio = module.rope_ratio if hasattr(module, "rope_ratio") else 1

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
        base = base * self.rope_ratio
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
        cache = torch.repeat_interleave(cache, 2, -2)
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len,
            self.dim,
            dtype=self.inv_freq.dtype,
            device=self.inv_freq.device,
        )


def GLM_prepare_inputs_for_generation(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    is_first_forward: bool = True,
    **kwargs,
) -> dict:
    if past_key_values is None:
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
        else:
            past_key_values = DynamicCache()
    # only last token for input_ids if past is not None
    if position_ids is None:
        position_ids = self.get_position_ids(attention_mask, device=input_ids.device)
    if not is_first_forward:
        if past_key_values.get_seq_length() != 0:
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


def chatglm_prepare_sdp_input(self, query, key, value, attention_mask, alibi):
    (
        dropout,
        alpha,
        beta,
        is_causal,
        blocked_attn_mask,
        blocked_alibi,
    ) = IPEXTransformerAttnOptimizedFp16.prepare_sdp_input(
        self, query, key, value, attention_mask, alibi
    )
    is_causal = True if self.is_1st_token() else False
    return dropout, alpha, beta, is_causal, blocked_attn_mask, blocked_alibi


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
        config.num_hidden_layers = config.num_layers
        self.ipex_config = self.build_ipex_transformer_config(
            config, device, dtype, impl_mode, tp_size, tp_group
        )
        grouped = True if self.ipex_config.multi_query_attention else False

        if dtype == "fp16":
            self.attn = (
                IPEXGroupedAttention(self.ipex_config, module.layer_number - 1)
                if grouped
                else IPEXAttention(self.ipex_config, module.layer_number - 1)
            )
        elif dtype == "int4" and xpu_gemm_use_xetla():
            self.attn = IPEXAttentionInt4(self.ipex_config, module.layer_number - 1)
        elif dtype == "int4" and not xpu_gemm_use_xetla():
            self.attn = IPEXAttentionInt4OneDNN(
                self.ipex_config, module.layer_number - 1
            )
        else:
            raise NotImplementedError(
                "IPEXAttention dose not support this modelType {} !".format(dtype)
            )

        self.mlp = self.build_mlp_from_config("ChatGLM")
        # self.attn.post_qkv = partial(chatglm_post_qkv, self.attn)
        # self.attn.prepare_sdp_input = partial(chatglm_prepare_sdp_input, self.attn)

        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
        )
        self.post_attn_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
        )

        self.port_all_parameters_to_new_module()
        self.glm_version = 3
        if "4" in config._name_or_path:
            self.glm_version = 4

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
            self.module.self_attention.query_key_value,
            self.module.self_attention.dense,
            dtype=self.ipex_config.dtype,
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
        dtype = self.ipex_config.dtype
        if self.ipex_config.multi_query_attention and dtype == "fp16":
            self.attn.transpose_parameter()
        else:
            self.attn.transpose_parameter(dtype=dtype)

        self.mlp.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        dtype = self.ipex_config.dtype
        if self.ipex_config.multi_query_attention:
            self.attn.load_parameter = partial(
                chatglm_load_attn_params_grouped, self.attn
            )
            if dtype == "int4":
                self.attn.transpose_parameter = partial(
                    transpose_attn_fused_qkv_params, self.attn
                )
        else:
            self.attn.load_parameter = partial(load_attn_fused_qkv_params, self.attn)
            self.attn.transpose_parameter = partial(
                transpose_attn_fused_qkv_params, self.attn
            )
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.LongTensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor]] = None,
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

        if self.glm_version == 3:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        b, seq, hidden_size = hidden_states.shape

        IPEXTransformerAttn.beam_size = b // bs
        beam = IPEXTransformerAttn.beam_size
        first_token = True if seq > 1 else False
        if first_token and beam > 1:
            hidden_states = hidden_states.view(bs, beam, seq, hidden_size)[
                :, 0, :, :
            ].contiguous()
            if rotary_pos_emb is not None:
                rotary_pos_emb = rotary_pos_emb.view(
                    bs,
                    beam,
                    rotary_pos_emb.shape[1],
                    rotary_pos_emb.shape[2],
                    rotary_pos_emb.shape[3],
                )[:, 0, :, :, :].view(
                    bs,
                    rotary_pos_emb.shape[1],
                    rotary_pos_emb.shape[2],
                    rotary_pos_emb.shape[3],
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

        if hasattr(kv_cache, "max_batch_size") and kv_cache.max_batch_size < b:
            repeat_cnt = b // kv_cache.max_batch_size
            for i in range(len(kv_cache.key_cache)):
                kv_cache.key_cache[i] = kv_cache.key_cache[i].repeat(
                    1, repeat_cnt, 1, 1
                )
                kv_cache.value_cache[i] = kv_cache.value_cache[i].repeat(
                    1, repeat_cnt, 1, 1
                )
            kv_cache.max_batch_size = b

        # seq first to be consist with query, key and value
        if (
            isinstance(kv_cache, IPEXStaticCache)
            and kv_cache.cache_format == CacheFormat.FBNH
            and not self.attn.beam_search_first_iter(seq)
            and rotary_pos_emb is not None
        ):
            rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

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
            past_key_value=kv_cache,
            use_cache=use_cache,
            residual=residual,
            cache_position=cache_position,
        )

        layernorm_output = self.post_attn_layernorm(layernorm_input)

        if self.config.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        mlp_output = self.mlp(layernorm_output, residual)

        if first_token and beam > 1:
            mlp_output = (
                mlp_output.view(bs, 1, seq, hidden_size)
                .expand([bs, beam, seq, hidden_size])
                .view(bs * beam, seq, hidden_size)
                .squeeze()
            )
        if self.glm_version == 3:
            mlp_output = mlp_output.transpose(0, 1)

        outputs = (mlp_output,)

        if use_cache:
            outputs += (present_key_value,)
        return outputs
