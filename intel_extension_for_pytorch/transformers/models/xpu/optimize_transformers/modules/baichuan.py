import torch
import math
from functools import partial
from typing import Optional, Tuple, List, Union
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

from .transformer_modules.Norm import LlamaRMSNorm
from .transformer_modules.CacheUtils import IPEXStaticCache, CacheFormat

from ._transformers import MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.Attention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16Baichuan,
)
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
from .transformer_modules.XPUAttentionfp16 import (
    IPEXAttention,
)
from .transformer_modules.XPUAttentionInt4 import (
    IPEXAttentionInt4,
    IPEXAttentionInt4OneDNN,
)


def BaichuanModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    output_hidden_states: Optional[bool] = False,
    return_dict: Optional[bool] = True,
) -> Union[Tuple, BaseModelOutputWithPast]:
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot provide both input_ids and inputs_embeds simultaneously"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You need to provide input_ids or inputs_embeds")

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    seq_length_with_past = seq_length

    if past_key_values is None:
        past_key_values_length = 0
    elif isinstance(past_key_values, Cache):
        past_key_values_length = past_key_values.get_seq_length()
    else:
        past_key_values_length = past_key_values[0][0].shape[2]
    seq_length_with_past = seq_length_with_past + past_key_values_length

    cache_position = torch.arange(
        past_key_values_length,
        past_key_values_length + seq_length,
        device=input_ids.device if input_ids is not None else inputs_embeds.device,
    )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self.training:
        if self.alibi_mask is None or self.alibi_mask.shape[-1] != seq_length_with_past:
            self.alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)
        alibi_mask = self.alibi_mask
    else:
        alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)

    if attention_mask is not None:
        if len(attention_mask.shape) == 2:
            expanded_mask = attention_mask.to(alibi_mask.dtype)
            expanded_mask = torch.tril(
                torch.gt(expanded_mask[:, :, None] * expanded_mask[:, None, :], 0)
            ) * torch.eq(expanded_mask[:, :, None] - expanded_mask[:, None, :], 0)
        else:
            expanded_mask = attention_mask
        bsz = inputs_embeds.size(0)
        src_len, tgt_len = alibi_mask.size()[-2:]
        expanded_mask = (
            expanded_mask.unsqueeze(1)
            .expand(bsz, 1, src_len, tgt_len)
            .to(alibi_mask.dtype)
        )
        inverted_mask = 1.0 - expanded_mask
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(alibi_mask.dtype).min
        )
        attention_mask = inverted_mask + alibi_mask.unsqueeze(0)
    else:
        attention_mask = alibi_mask

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                layer_idx=idx,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def Baichuan_prepare_inputs_for_generation(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    **kwargs,
):
    if past_key_values is None:
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
        else:
            past_key_values = DynamicCache()
    if past_key_values.get_seq_length() != 0:
        input_ids = input_ids[:, -1:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values.get_seq_length() == 0:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


def baichuan_load_parameter(self, qkv_proj, out_proj):
    self.qkv_proj.weight = qkv_proj.weight
    self.out_proj.weight = out_proj.weight

    self.qkv_proj.bias = qkv_proj.bias
    self.out_proj.bias = out_proj.bias


def baichuan_transpose_parameter(self):
    self.qkv_proj.weight.data = (
        self.qkv_proj.weight.view(3, self.num_attn_head * self.head_dim, self.embed_dim)
        .permute(0, 2, 1)
        .contiguous()
    )
    self.out_proj.weight.data = self.out_proj.weight.transpose(0, 1).contiguous()
    # Note: synchronize to ensure the completion of contiguous
    torch.xpu.synchronize()


def sdp(self, query, key, value, past_key_value, attention_mask, head_mask, alibi):
    scale = 1.0 / math.sqrt(self.head_dim)
    use_casual = False

    # we are not plan to support attention mask here, for it should be in None
    # at all of our test case.
    if attention_mask is not None:
        if query.size()[2] == 1:  # inference with cache
            if len(attention_mask.size()) == 4:
                attention_mask = attention_mask[:, :, -1:, :]
            else:
                attention_mask = attention_mask[:, -1:, :]

    if attention_mask is not None:
        attention_mask = self.get_blocked_attn_mask(attention_mask)
    if alibi is not None:
        alibi = self.get_blocked_alibi(alibi, key.size(1))
    if (
        self.beam_idx is not None
        and query.size(-2) == 1
        and isinstance(past_key_value, IPEXStaticCache)
    ):
        use_casual = False
        key_prompt, value_prompt = past_key_value.get_prompt_for_beam_search(
            self.layer_idx
        )
        prompt_length = key_prompt.size(2)
        seqlen = key.size(2)
        # TODO: remove this after ifmha support combined kv cache with both prompt
        # and decode in [bs, seqlen, num_head, head_dim] layout
        key = key[:, :, prompt_length:, :]
        value = value[:, :, prompt_length:, :]
        # TODO: remove this after ifmha support [bs, seqlen, num_head, head_dim] layout
        if (
            isinstance(past_key_value, IPEXStaticCache)
            and past_key_value.cache_format == CacheFormat.BFNH
        ):  # for BFNH format
            key = key.permute(2, 0, 1, 3).contiguous().permute(1, 2, 0, 3)
            value = (
                value.permute(2, 0, 1, 3)
                .contiguous()
                .permute(
                    1,
                    2,
                    0,
                )
            )

        attention_output = torch.xpu.IpexSDP_Index(
            query,
            key_prompt,
            value_prompt,
            key,
            value,
            self.beam_idx,
            alibi,
            attention_mask,
            head_mask,
            seqlen,
            scale,
            1.0,
            0.0,
            use_casual,
        )
    else:
        # TODO: remove this after fmha support strided fmha on F dim
        if (
            isinstance(past_key_value, IPEXStaticCache)
            and past_key_value.cache_format == CacheFormat.FBNH
        ):  # for BFNH format
            attention_output = torch.xpu.IpexSDP(
                query,
                key,
                value,
                alibi,
                attention_mask,
                head_mask,
                scale,
                1.0,
                0.0,
                use_casual,
                self.beam_idx is None,
            )
        else:  # for BFNH format
            # TODO: remove this after fmha support strided fmha on F dim
            if query.size(0) > 1:
                key = key.transpose(1, 2).contiguous().transpose(1, 2)
                value = value.transpose(1, 2).contiguous().transpose(1, 2)

            attention_output = torch.xpu.IpexSDP(
                query,
                key,
                value,
                alibi,
                attention_mask,
                head_mask,
                scale,
                1.0,
                0.0,
                use_casual,
                False,
            )

    return attention_output, None


class NewIPEXBaichuanBlock(IPEXTransformerBlock):
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
        if dtype == "fp16":
            self.attn = IPEXAttention(self.ipex_config)
        elif dtype == "int4" and xpu_gemm_use_xetla():
            self.attn = IPEXAttentionInt4(self.ipex_config)
        elif dtype == "int4" and not xpu_gemm_use_xetla():
            self.attn = IPEXAttentionInt4OneDNN(self.ipex_config)
        else:
            raise NotImplementedError(
                "IPEXAttention dose not support this modelType {} !".format(dtype)
            )
        self.mlp = self.build_mlp_from_config("Baichuan")

        self.attn.load_parameter = partial(baichuan_load_parameter, self.attn)
        self.attn.transpose_parameter = partial(baichuan_transpose_parameter, self.attn)
        self.attn.sdp = partial(sdp, self.attn)

        self.input_layernorm = LlamaRMSNorm(
            self.ipex_config.embedding_dim, self.ipex_config.norm_eps
        )
        self.post_attn_layernorm = LlamaRMSNorm(
            self.ipex_config.embedding_dim, self.ipex_config.norm_eps
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
            # transformers==4.31.0
            num_key_value_head=self.config.num_attention_heads,
            max_positions=max(
                (
                    self.config.max_position_embeddings
                    if hasattr(self.config, "max_position_embeddings")
                    else self.config.model_max_length
                ),
                MAX_SEQ_LEN,
            ),
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=None,
            rotary_dim=None,
            rotary_half=True,
            rotate_every_two=False,
            use_causal_mask=False,
            activation_function=self.config.hidden_act,
            ipex_act=ipex_activation,
            norm_eps=self.config.rms_norm_eps,
            residual_dropout=None,
            attn_dropout=None,
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
        # IPEXTransformerAttnOptimizedFp16Baichuan
        self.attn.load_parameter(
            self.module.self_attn.W_pack,
            self.module.self_attn.o_proj,
        )

    def port_mlp_parameter(self):
        # IPEXTransformerMLPOptimizedFp16SiluBaichuan
        self.mlp.load_parameter(
            self.module.mlp.gate_proj,
            self.module.mlp.down_proj,
            self.module.mlp.up_proj,
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        layer_idx: Optional[int] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        bs = IPEXTransformerAttn.batch_size
        IPEXTransformerAttn.beam_size = hidden_states.shape[0] // bs
        beam = IPEXTransformerAttn.beam_size

        # broadcast attention mask if needed
        if attention_mask.dim() < 4:
            attention_mask = (
                attention_mask.unsqueeze(0)
                .expand(
                    bs * beam,
                    attention_mask.shape[0],
                    attention_mask.shape[1],
                    attention_mask.shape[2],
                )
                .contiguous()
            )

        _, seq, hidden_size = hidden_states.shape
        first_token = True if seq > 1 else False
        if first_token and beam > 1:
            hidden_states = hidden_states.view(bs, beam, seq, hidden_size)[
                :, 0, :, :
            ].contiguous()
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
        if (
            hasattr(past_key_value, "max_batch_size")
            and past_key_value.max_batch_size < beam
        ):
            repeat_cnt = beam // past_key_value.max_batch_size
            for i in range(len(past_key_value.key_cache)):
                past_key_value.key_cache[i] = past_key_value.key_cache[i].repeat(
                    1, repeat_cnt, 1, 1
                )
                past_key_value.value_cache[i] = past_key_value.value_cache[i].repeat(
                    1, repeat_cnt, 1, 1
                )
            past_key_value.max_batch_size = beam

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        self.attn.layer_idx = layer_idx

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
            )
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        return outputs
