from typing import Optional, Tuple, Union
from functools import partial
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.activations import get_activation
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from ._transformers import MAX_OUT_SEQ_LEN, MAX_SEQ_LEN
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.DecoderBlock import IPEXTransformerBlock
from .transformer_modules.GroupedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16Grouped,
)
from .transformer_modules.Mlp import *  # noqa
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .transformer_modules.QuantizedAttention import (  # noqa F401; noqa
    IPEXTransformerAttnOptimizedFp16,
)
from .transformer_modules.RoPE import LlamaRotaryEmbedding
from .transformer_modules.XPUAttentionfp16 import (
    IPEXAttention,
    IPEXGroupedAttention,
)

from .transformer_modules.model_utils import (
    load_attn_fused_qkv_params,
    chatglm_load_attn_params_grouped,
    transpose_attn_fused_qkv_params,
)


def dropout_add(
    x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool
) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`):
            input tensor
        residual (`torch.tensor`):
            residual tensor
        prob (`float`):
            dropout probability
        training (`bool`):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class FalconLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_states = input @ self.weight.T
        if self.bias is None:
            return hidden_states
        return hidden_states + self.bias


class FalconMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size

        self.dense_h_to_4h = FalconLinear(
            hidden_size, config.ffn_hidden_size, bias=config.bias
        )
        self.act = get_activation(config.activation)
        self.dense_4h_to_h = FalconLinear(
            config.ffn_hidden_size, hidden_size, bias=config.bias
        )
        self.hidden_dropout = config.hidden_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x


class NewIPEXFalconBlock(IPEXTransformerBlock):
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
        self.new_decoder_architecture = (
            True if config.new_decoder_architecture else False
        )
        self.num_ln_in_parallel_attn = (
            2
            if config.num_ln_in_parallel_attn is None and self.new_decoder_architecture
            else config.num_ln_in_parallel_attn
        )
        self.parallel_attn = config.parallel_attn
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(
            config, device, dtype, impl_mode, tp_size, tp_group
        )

        self.grouped = False
        if dtype == "fp16":
            if (
                self.ipex_config.num_attention_head
                > self.ipex_config.num_key_value_head
            ):
                self.grouped = True
                self.attn = IPEXGroupedAttention(self.ipex_config)
            else:
                self.attn = IPEXAttention(self.ipex_config)
        else:
            raise NotImplementedError(
                "IPEXAttention dose not support this modelType {} !".format(dtype)
            )

        if not self.new_decoder_architecture:
            if self.grouped:
                self.attn.load_parameter = partial(
                    chatglm_load_attn_params_grouped, self.attn
                )
            else:
                self.attn.load_parameter = partial(
                    load_attn_fused_qkv_params, self.attn
                )
                self.attn.transpose_parameter = partial(
                    transpose_attn_fused_qkv_params, self.attn
                )

        self.mlp = (
            FalconMLP(config)
            if not self.new_decoder_architecture
            else self.build_mlp_from_config("Falcon")
        )

        if not self.parallel_attn:
            self.post_attention_layernorm = nn.LayerNorm(
                self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
            )
            self.input_layernorm = nn.LayerNorm(
                self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
            )
        else:
            if self.num_ln_in_parallel_attn == 2:
                self.ln_attn = nn.LayerNorm(
                    self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
                )
                self.ln_mlp = nn.LayerNorm(
                    self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
                )
            else:
                self.input_layernorm = nn.LayerNorm(
                    self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
                )

        self.port_all_parameters_to_new_module()

    def build_ipex_transformer_config(
        self, config, device, dtype, impl_mode, tp_size, tp_group
    ) -> IPEXTransformerConfig:
        activation_function = "gelu"
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
            intermediate_dim=self.config.hidden_size * 4,
            num_attention_head=self.config.num_attention_heads,
            num_key_value_head=self.config.num_key_value_heads,
            max_positions=max(2048, MAX_SEQ_LEN),
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=LlamaRotaryEmbedding,
            rotary_dim=None,
            rotary_half=True,
            rotate_every_two=False,
            use_causal_mask=False,
            activation_function=activation_function,
            ipex_act=ipex_activation,
            norm_eps=self.config.layer_norm_epsilon,
            residual_dropout=self.config.hidden_dropout,
            attn_dropout=self.config.attention_dropout,
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
        if self.new_decoder_architecture:
            self.attn.load_parameter(
                qkv_proj=self.module.self_attention.query_key_value,
                out_proj=self.module.self_attention.dense,
            )
        else:
            self.attn.load_parameter(
                self.module.self_attention.query_key_value,
                self.module.self_attention.dense,
                dtype=self.ipex_config.dtype,
            )

    def port_mlp_parameter(self):
        if self.new_decoder_architecture:
            self.mlp.load_parameter(
                self.module.mlp.dense_h_to_4h, self.module.mlp.dense_4h_to_h
            )
        else:
            self.mlp.dense_h_to_4h = self.module.mlp.dense_h_to_4h
            self.mlp.dense_4h_to_h = self.module.mlp.dense_4h_to_h

    def port_norm_parameter(self):
        if not self.parallel_attn:
            self.post_attention_layernorm.weight = (
                self.module.post_attention_layernorm.weight
            )
            self.post_attention_layernorm.bias = (
                self.module.post_attention_layernorm.bias
            )
            self.input_layernorm.weight = self.module.input_layernorm.weight
            self.input_layernorm.bias = self.module.input_layernorm.bias
        else:
            if self.num_ln_in_parallel_attn == 2:
                self.ln_attn.weight = self.module.ln_attn.weight
                self.ln_attn.bias = self.module.ln_attn.bias
                self.ln_mlp.weight = self.module.ln_mlp.weight
                self.ln_mlp.bias = self.module.ln_mlp.bias
            else:
                self.input_layernorm.weight = self.module.input_layernorm.weight
                self.input_layernorm.bias = self.module.input_layernorm.bias

    def transpose_parameter(self):
        if self.new_decoder_architecture:
            self.attn.transpose_parameter()
            self.mlp.transpose_parameter()
        else:
            if not self.grouped:
                dtype = self.ipex_config.dtype
                self.attn.transpose_parameter(dtype=dtype)
            else:
                self.attn.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()
        self.attn.cat_qkv()

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        layer_idx: int = 0,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        bs = IPEXTransformerAttn.batch_size
        IPEXTransformerAttn.beam_size = hidden_states.shape[0] // bs

        if layer_past.get_seq_length() and len(layer_past.key_cache) > layer_idx:
            if layer_past.key_cache[layer_idx].dim() == 3:
                key_cache = layer_past.key_cache[layer_idx]
                value_cache = layer_past.value_cache[layer_idx]
                batch_size_times_num_heads, kv_length, head_dim = key.cache.shape
                num_heads = batch_size_times_num_heads // bs
                key_cache = key_cache.view(bs, num_heads, kv_length, head_dim)
                value_cache = value_cache.view(bs, num_heads, kv_length, head_dim)
                layer_past.key_cache[layer_idx] = key_cache
                layer_past.value_cache[layer_idx] = value_cache
        residual = hidden_states
        if self.new_decoder_architecture and self.num_ln_in_parallel_attn == 2:
            attention_layernorm_out = torch.ops.torch_ipex.fast_layer_norm(
                hidden_states,
                self.ln_attn.normalized_shape,
                self.ln_attn.weight,
                self.ln_attn.bias,
                self.ln_attn.eps,
            )
            mlp_layernorm_out = torch.ops.torch_ipex.fast_layer_norm(
                hidden_states,
                self.ln_mlp.normalized_shape,
                self.ln_mlp.weight,
                self.ln_mlp.bias,
                self.ln_mlp.eps,
            )
        else:
            attention_layernorm_out = torch.ops.torch_ipex.fast_layer_norm(
                hidden_states,
                self.input_layernorm.normalized_shape,
                self.input_layernorm.weight,
                self.input_layernorm.bias,
                self.input_layernorm.eps,
            )
        attn_outputs = self.attn(
            hidden_states=attention_layernorm_out,
            past_key_value=layer_past,
            attention_mask=None,
            cache_position=cache_position,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            alibi=alibi,
            **kwargs,
        )

        attention_output = attn_outputs[0]
        if not self.new_decoder_architecture:
            if self.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(
                    attention_output,
                    residual,
                    self.ipex_config.attn_dropout,
                    training=self.training,
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        if (
            self.new_decoder_architecture
            and self.parallel_attn
            and self.num_ln_in_parallel_attn == 1
        ):
            mlp_layernorm_out = attention_layernorm_out
        outputs = attn_outputs[1:]

        # residual is already fused into attention
        mlp_output = self.mlp(mlp_layernorm_out)
        if self.new_decoder_architecture or self.parallel_attn:
            mlp_output += attention_output
        output = dropout_add(
            mlp_output,
            residual,
            self.ipex_config.residual_dropout,
            training=self.training,
        )
        next_cache = None
        if use_cache:
            layer_past = outputs[0]
            outputs = (output, layer_past) + outputs[1:]
        else:
            outputs = (output,) + outputs[1:]
        return outputs


def IPEXFalconModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
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

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if past_key_values is None:
        past_key_values = tuple([None] * len(self.h))

    use_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache) and not self.training:
        use_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False
    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    past_key_values_length = 0
    # Compute alibi tensor: check build_alibi_tensor documentation
    if isinstance(past_key_values, Cache) and past_key_values.get_seq_length() > 0:
        past_key_values_length = past_key_values.get_seq_length()

    if self.use_alibi:
        mask = (
            torch.ones(
                (batch_size, seq_length + past_key_values_length),
                device=inputs_embeds.device,
                dtype=torch.long,
            )
            if attention_mask is None
            else attention_mask
        )
        alibi = build_alibi_tensor(mask, self.num_heads, dtype=hidden_states.dtype)
    else:
        alibi = None
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

    if cache_position is None:
        past_seen_tokens = past_key_values_length
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        if alibi is None:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        elif head_mask is None:
            alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])

            # We don't call _prepare_4d_causal_attention_mask_for_sdpa as we
            # need to mask alibi using the 4D attention_mask untouched.
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

            # We take care to integrate alibi bias in the attention_mask here.
            min_dtype = torch.finfo(alibi.dtype).min
            attention_mask = torch.masked_fill(
                alibi / math.sqrt(self.config.hidden_size // self.num_heads),
                attention_mask < -1,
                min_dtype,
            )

            # From PyTorch 2.1 onwards, F.scaled_dot_product_attention with the memory-efficient attention backend
            # produces nans if sequences are completely unattended in the attention mask.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            if seq_length > 1 and attention_mask.device.type == "cuda":
                attention_mask = AttentionMaskConverter._unmask_unattended(
                    attention_mask, min_dtype=min_dtype
                )
        else:
            # PyTorch SDPA does not support head_mask, we fall back on the eager implementation in this case.
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape batch_size x num_heads x N x N
    # head_mask has shape n_layer x batch x num_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    for i, block in enumerate(self.h):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                block.__call__,
                hidden_states,
                alibi,
                attention_mask,
                position_ids,
                head_mask[i],
                layer_past,
                use_cache,
                output_attentions,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=past_key_values,
                layer_idx=i,
                attention_mask=attention_mask,
                cache_position=cache_position,
                position_ids=position_ids,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = outputs[1]

        if output_attentions:
            all_self_attentions = all_self_attentions + (
                outputs[2 if use_cache else 1],
            )

    # Add last hidden state
    hidden_states = self.ln_f(hidden_states)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = presents.to_legacy_cache() if use_legacy_cache else presents
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def Falcon_prepare_inputs_for_generation(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    if isinstance(past_key_values, Cache) and past_key_values.get_seq_length():
        past_length = past_key_values.get_seq_length()

        # Some generation methods already pass only the last input ID
        if input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            # Default to old behavior: keep only final ID
            remove_prefix_length = input_ids.shape[1] - 1
        input_ids = input_ids[:, remove_prefix_length:]

    # Note: versions of Falcon with alibi do not use position_ids. It is used with RoPE.
    if (
        not self.transformer.use_alibi
        and attention_mask is not None
        and position_ids is None
    ):
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        # The clone here is for the same reason as for `position_ids`.
        model_inputs = {
            "input_ids": input_ids.clone(memory_format=torch.contiguous_format),
            "inputs_embeds": None,
        }

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
