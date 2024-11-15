import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.cache_utils import Cache, DynamicCache
from .transformer_modules.Attention import IPEXTransformerAttnOptimizedFp16  # noqa

from .transformer_modules.RoPE import PositionalEmbedding

from ._transformers import MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.Mlp import (  # noqa F401
    IPEXTransformerBaseMLP,
    IPEXTransformerMLPOptimizedFp16,
)  # noqa
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from .transformer_modules.QuantizedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedInt4,
)  # noqa
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .transformer_modules.CrossedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16Crossed,
)  # noqa
from .transformer_modules.DecoderBlock import IPEXTransformerBlock
from .transformer_modules.Mlp import *  # noqa
from .transformer_modules.XPUAttentionfp16 import (
    IPEXAttention,
)
from .transformer_modules.XPUAttentionInt4 import (
    IPEXAttentionInt4,
    IPEXAttentionInt4OneDNN,
)


class NewIPEXOPTBlock(IPEXTransformerBlock):
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
            self.self_attn = IPEXAttention(self.ipex_config)
        elif dtype == "int4" and xpu_gemm_use_xetla():
            self.self_attn = IPEXAttentionInt4(self.ipex_config)
        elif dtype == "int4" and not xpu_gemm_use_xetla():
            self.self_attn = IPEXAttentionInt4OneDNN(self.ipex_config)
        else:
            raise NotImplementedError(
                "IPEXAttention dose not support this modelType {} !".format(dtype)
            )
        self.mlp = self.build_mlp_from_config("Opt")
        self.do_layer_norm_before = self.ipex_config.do_norm_before
        self.self_attn_layer_norm = nn.LayerNorm(
            self.ipex_config.embedding_dim,
            elementwise_affine=self.ipex_config.ln_elementwise_affine,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.ipex_config.embedding_dim,
            elementwise_affine=self.ipex_config.ln_elementwise_affine,
        )
        self.dropout_p = self.ipex_config.residual_pdrop
        self.port_all_parameters_to_new_module()

    def build_ipex_transformer_config(
        self, config, device, dtype, impl_mode, tp_size, tp_group
    ) -> IPEXTransformerConfig:
        activation_function = self.config.activation_function
        ipex_activation = None
        for act in SupportedActivation:
            if activation_function in act.value:
                ipex_activation = act
                break
        assert ipex_activation is not None, (
            "found unrecognized activation "
            "function, can not build ipex config from {}".format(activation_function)
        )

        assert dtype in [
            "fp16",
            "int4",
        ], "dtype tag {} passed to " "optimized_transformers is not supported!".format(
            dtype
        )

        return IPEXTransformerConfig(
            embedding_dim=self.config.hidden_size,
            intermediate_dim=self.config.ffn_dim,
            num_attention_head=self.config.num_attention_heads,
            num_key_value_head=self.config.num_attention_heads,
            max_positions=max(self.config.max_position_embeddings, MAX_SEQ_LEN),
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=PositionalEmbedding,
            rotary_dim=None,
            rotary_half=False,
            rotate_every_two=False,
            use_causal_mask=False,
            activation_function=self.config.activation_function,
            ipex_act=ipex_activation,
            norm_eps=None,
            residual_dropout=self.config.dropout,
            attn_dropout=None,
            enable_bias=self.config.enable_bias,
            residual_pdrop=self.config.dropout,
            scale_attention=True,
            is_decoder=True,
            do_norm_before=self.config.do_layer_norm_before,
            ln_elementwise_affine=self.config.layer_norm_elementwise_affine,
            positional_embedding_base=10000,
            device=self.device,
            dtype=dtype,
            impl=impl_mode,
            tp_size=tp_size,
            tp_group=tp_group,
        )

    def port_attn_parameter(self):
        # IPEXTransformerAttnOptimizedFp16Opt IPEXTransformerAttnOptimizedFp16
        self.self_attn.load_parameter(
            self.module.self_attn.q_proj,
            self.module.self_attn.k_proj,
            self.module.self_attn.v_proj,
            self.module.self_attn.out_proj,
        )

    def port_mlp_parameter(self):
        # IPEXTransformerMLPOptimizedFp16ReluOpt IPEXTransformerAttnOptimizedFp16
        self.mlp.load_parameter(self.module.fc1, self.module.fc2)

    def port_norm_parameter(self):
        self.self_attn_layer_norm.weight = self.module.self_attn_layer_norm.weight
        self.self_attn_layer_norm.bias = self.module.self_attn_layer_norm.bias
        self.final_layer_norm.weight = self.module.final_layer_norm.weight
        self.final_layer_norm.bias = self.module.final_layer_norm.bias

    def transpose_parameter(self):
        self.self_attn.transpose_parameter()
        self.mlp.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()
        self.self_attn.cat_qkv()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[List[torch.FloatTensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        bs = IPEXTransformerAttn.batch_size
        IPEXTransformerAttn.beam_size = hidden_states.shape[0] // bs

        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = torch.ops.torch_ipex.fast_layer_norm(
                hidden_states,
                self.self_attn_layer_norm.normalized_shape,
                self.self_attn_layer_norm.weight,
                self.self_attn_layer_norm.bias,
                self.self_attn_layer_norm.eps,
            )

        hidden_states, present_key_value, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            head_mask=layer_head_mask,
            output_attentions=output_attentions,
            residual=residual,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )

        if not self.do_layer_norm_before:
            hidden_states = torch.ops.torch_ipex.fast_layer_norm(
                hidden_states,
                self.self_attn_layer_norm.normalized_shape,
                self.self_attn_layer_norm.weight,
                self.self_attn_layer_norm.bias,
                self.self_attn_layer_norm.eps,
            )

        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = torch.ops.torch_ipex.fast_layer_norm(
                hidden_states,
                self.final_layer_norm.normalized_shape,
                self.final_layer_norm.weight,
                self.final_layer_norm.bias,
                self.final_layer_norm.eps,
            )
        hidden_states = self.mlp(hidden_states, residual)
        if not self.do_layer_norm_before:
            hidden_states = torch.ops.torch_ipex.fast_layer_norm(
                hidden_states,
                self.final_layer_norm.normalized_shape,
                self.final_layer_norm.weight,
                self.final_layer_norm.bias,
                self.final_layer_norm.eps,
            )

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


def IPEXOPTDecoder_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    use_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache) and not self.training:
        use_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

    batch_size, seq_length = input_shape
    if isinstance(past_key_values, Cache) and past_key_values.get_seq_length() > 0:
        past_key_values_length = past_key_values.get_seq_length()
    elif isinstance(past_key_values, Tuple):
        past_key_values_length = past_key_values[0][0].shape[2]
    else:
        past_key_values_length = 0

    # required mask seq length can be calculated via length of past
    mask_seq_length = past_key_values_length + seq_length

    if cache_position is None:
        past_seen_tokens = past_key_values_length
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # embed positions
    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        causal_attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
        attention_mask = (
            torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
            if attention_mask is None
            else attention_mask
        )
    else:
        # 4d mask is passed through the layers
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, mask_seq_length, device=inputs_embeds.device
            )
        elif attention_mask.shape[1] != mask_seq_length:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{mask_seq_length} (sum of the lengths of current and past inputs)"
            )
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

    pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

    if self.project_in is not None:
        inputs_embeds = self.project_in(inputs_embeds)

    hidden_states = inputs_embeds + pos_embeds

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

    # check if head_mask has a correct number of layers specified if desired
    for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
        if attn_mask is not None:
            if attn_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

    for idx, decoder_layer in enumerate(self.layers):
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:
                continue
        if past_key_values is not None and isinstance(past_key_values, Tuple):
            past_key_value = past_key_values[idx]
        elif (
            isinstance(past_key_values, Cache) and past_key_values.get_seq_length() > 0
        ):
            past_key_value = past_key_values
        else:
            past_key_value = None
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_attention_mask,
                head_mask[idx] if head_mask is not None else None,
                None,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_ids=position_ids,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    if self.final_layer_norm is not None:
        hidden_states = self.final_layer_norm(hidden_states)

    if self.project_out is not None:
        hidden_states = self.project_out(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )
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


def OPT_prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    use_cache=True,
    **kwargs,
):
    if isinstance(past_key_values, Cache) and past_key_values.get_seq_length() > 0:
        past_length = past_key_values.get_seq_length()

        # Some generation methods already pass only the last input ID
        if input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            # Default to old behavior: keep only final ID
            remove_prefix_length = input_ids.shape[1] - 1

        input_ids = input_ids[:, remove_prefix_length:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
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
