import torch
from torch import nn
from typing import Optional, Tuple, Union, List
from ...reference.fusions.linear_fusion import (
    _IPEXlinearAddRef,
    _IPEXlinearAddAddRef,
    _IPEXlinearNewGeluRef,
    _IPEXlinearReluRef,
    _IPEXlinearGeluRef,
    _IPEXlinearMulRef,
    _IPEXlinearSiluMulRef,
)
from torch.nn import functional as F
from .....utils._logger import logger, WarningType


def LlamaDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    if not self.distributed:
        hidden_states = self.mha_linear_add(hidden_states, residual)
    else:
        hidden_states = self.self_attn.o_proj(hidden_states)
        hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    mlp_gate = self.linear_silu_mul(hidden_states)

    if not self.distributed:
        hidden_states = self.mlp_linear_add(mlp_gate, residual)
    else:
        hidden_states = self.mlp.down_proj(mlp_gate)
        hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def OPTDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
    if self.do_layer_norm_before:
        hidden_states = self.self_attn_layer_norm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=past_key_value,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
    )
    if not self.distributed:
        hidden_states = self.mha_linear_add(hidden_states, residual)
    else:
        hidden_states = self.self_attn.out_proj(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
    # 350m applies layer norm AFTER attention
    if not self.do_layer_norm_before:
        hidden_states = self.self_attn_layer_norm(hidden_states)

    # Fully Connected
    hidden_states_shape = hidden_states.shape
    residual = hidden_states

    # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
    if self.do_layer_norm_before:
        hidden_states = self.final_layer_norm(hidden_states)

    hidden_states = self.linear_relu(hidden_states)

    if not self.distributed:
        hidden_states = self.mlp_linear_add(hidden_states, residual).view(
            hidden_states_shape
        )
    else:
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = (residual + hidden_states).view(hidden_states_shape)

    # 350m applies layer norm AFTER attention
    if not self.do_layer_norm_before:
        hidden_states = self.final_layer_norm(hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def GPTJBlock_forward(
    self,
    hidden_states: Optional[torch.FloatTensor],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]
]:
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(
        hidden_states=hidden_states,
        layer_past=layer_past,
        attention_mask=attention_mask,
        position_ids=position_ids,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_outputs[1:]

    feed_forward_hidden_states = self.linear_gelu(hidden_states)

    if not self.distributed:
        hidden_states = self.linear_add_add(
            feed_forward_hidden_states, attn_output, residual
        )
    else:
        feed_forward_hidden_states = self.mlp.fc_out(feed_forward_hidden_states)
        feed_forward_hidden_states = self.mlp.dropout(feed_forward_hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions)


def FalconDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    alibi: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
):
    residual = hidden_states
    if (
        self.self_attention.new_decoder_architecture
        and not (
            hasattr(self.config, "num_ln_in_parallel_attn")
            and self.config.num_ln_in_parallel_attn == 1
        )
    ) or not hasattr(self, "input_layernorm"):
        attention_layernorm_out = self.ln_attn(hidden_states)
        mlp_layernorm_out = self.ln_mlp(hidden_states)
    else:
        attention_layernorm_out = self.input_layernorm(hidden_states)
    # Self attention.
    attn_outputs = self.self_attention(
        attention_layernorm_out,
        layer_past=layer_past,
        attention_mask=attention_mask,
        alibi=alibi,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    attention_output = attn_outputs[0]
    if not (
        self.self_attention.new_decoder_architecture
        or not hasattr(self, "input_layernorm")
    ):
        if self.config.parallel_attn:
            mlp_layernorm_out = attention_layernorm_out
        else:
            residual = attention_output + residual
            mlp_layernorm_out = self.post_attention_layernorm(residual)
    if (
        self.config.new_decoder_architecture
        and self.config.parallel_attn
        and hasattr(self.config, "num_ln_in_parallel_attn")
        and self.config.num_ln_in_parallel_attn == 1
    ):
        mlp_layernorm_out = attention_layernorm_out
    outputs = attn_outputs[1:]
    # MLP.

    mlp_hidden_states = self.linear_gelu(mlp_layernorm_out)
    if not self.distributed:
        if (
            self.self_attention.new_decoder_architecture
            or self.config.parallel_attn
            or not hasattr(self, "input_layernorm")
        ):
            output = self.linear_add_add(mlp_hidden_states, attention_output, residual)
        else:
            output = self.linear_add(mlp_hidden_states, residual)
    else:
        mlp_output = self.mlp.dense_4h_to_h(mlp_hidden_states)
        if (
            self.self_attention.new_decoder_architecture
            or self.config.parallel_attn
            or not hasattr(self, "input_layernorm")
        ):
            mlp_output += attention_output
        output = mlp_output + residual

    if use_cache:
        outputs = (output,) + outputs
    else:
        outputs = (output,) + outputs[1:]
    return outputs  # hidden_states, present, attentions


def BloomBlock_forward(
    self,
    hidden_states: torch.Tensor,
    alibi: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
):
    # hidden_states: [batch_size, seq_length, hidden_size]

    # Layer norm at the beginning of the transformer layer.
    layernorm_output = self.input_layernorm(hidden_states)

    # Layer norm post the self attention.
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = hidden_states

    # Self attention.
    attn_outputs = self.self_attention(
        layernorm_output,
        residual=residual,
        layer_past=layer_past,
        attention_mask=attention_mask,
        alibi=alibi,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )

    attention_output = attn_outputs[0]

    outputs = attn_outputs[1:]

    layernorm_output = self.post_attention_layernorm(attention_output)

    # Get residual
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = attention_output

    feed_forward_hidden_states = self.linear_gelu(layernorm_output)
    if not self.distributed:
        output = self.linear_add(feed_forward_hidden_states, residual)
    else:
        intermediate_output = self.mlp.dense_4h_to_h(feed_forward_hidden_states)
        output = intermediate_output + residual

    if use_cache:
        outputs = (output,) + outputs
    else:
        outputs = (output,) + outputs[1:]

    return outputs  # hidden_states, present, attentions


def BaichuanDecoder_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    if not self.distributed:
        hidden_states = self.mha_linear_add(hidden_states, residual)
    else:
        hidden_states = self.self_attn.o_proj(hidden_states)
        hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    mlp_gate = self.linear_silu_mul(hidden_states)
    if not self.distributed:
        hidden_states = self.mlp_linear_add(mlp_gate, residual)
    else:
        hidden_states = self.mlp.down_proj(mlp_gate)
        hidden_states = residual + hidden_states
    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def GLMBlock_forward(
    self,
    hidden_states,
    attention_mask,
    rotary_pos_emb,
    kv_caches=None,
    use_cache=True,
):
    # hidden_states: [s, b, h]

    # Layer norm at the beginning of the transformer layer.
    layernorm_output = self.input_layernorm(hidden_states)
    # Self attention.
    attention_output, kv_cache = self.self_attention(
        hidden_states=layernorm_output,
        attention_mask=attention_mask,
        rotary_pos_emb=rotary_pos_emb,
        kv_caches=kv_caches,
        use_cache=use_cache,
    )

    # Residual connection.
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = hidden_states

    if not self.distributed:
        layernorm_input = self.mha_linear_add(attention_output, residual)
    else:
        hidden_states = self.self_attention.dense(attention_output)
        layernorm_input = residual + hidden_states

    # Layer norm post the self attention.
    layernorm_output = self.post_attention_layernorm(layernorm_input)

    # # MLP.
    if hasattr(self, "linear_silu_mul"):
        intermediate_parallel = self.linear_silu_mul(layernorm_output)
    else:
        intermediate_parallel = self.mlp.dense_h_to_4h(layernorm_output)
        intermediate_parallel = self.mlp.activation_func(intermediate_parallel)

    # Second residual connection.
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = layernorm_input

    if not self.distributed:
        output = self.mlp_linear_add(intermediate_parallel, residual)
    else:
        hidden_states = self.mlp.dense_4h_to_h(intermediate_parallel)
        output = residual + hidden_states

    return output, kv_cache


def GPTBigCodeBlock_forward(
    self,
    hidden_states: Optional[Tuple[torch.Tensor]],
    layer_past: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(
        hidden_states,
        layer_past=layer_past,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_outputs[1:]
    # residual connection
    if not self.distributed:
        hidden_states = self.mha_linear_add(attn_output, residual)
    else:
        attn_output = self.attn.c_proj(attn_output)
        hidden_states = attn_output + residual

    if encoder_hidden_states is not None:
        # add one self-attention block for cross-attention
        if not hasattr(self, "crossattention"):
            raise ValueError(
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                "cross-attention layers by setting `config.add_cross_attention=True`"
            )
        residual = hidden_states
        hidden_states = self.ln_cross_attn(hidden_states)
        cross_attn_outputs = self.crossattention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = cross_attn_outputs[0]
        # residual connection
        hidden_states = residual + attn_output
        outputs = (
            outputs + cross_attn_outputs[2:]
        )  # add cross attentions if we output attention weights

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    feed_forward_hidden_states = self.linear_gelu(hidden_states)
    if not self.distributed:
        hidden_states = self.mlp_linear_add(feed_forward_hidden_states, residual)
    else:
        feed_forward_hidden_states = self.mlp.c_proj(feed_forward_hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions, cross_attentions)


def T5Block_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_bias=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    encoder_decoder_position_bias=None,
    layer_head_mask=None,
    cross_attn_layer_head_mask=None,
    past_key_value=None,
    use_cache=False,
    output_attentions=False,
    return_dict=True,
):
    if past_key_value is not None:
        if not self.is_decoder:
            logger.warning(
                "`past_key_values` is passed to the encoder. Please make sure this is intended."
            )
        expected_num_past_key_values = 4 if encoder_hidden_states is None else 8

        self_attn_past_key_value = past_key_value[:4]
        if len(past_key_value) != expected_num_past_key_values:
            cross_attn_past_key_value = None
        else:
            cross_attn_past_key_value = past_key_value[4:]
    else:
        self_attn_past_key_value, cross_attn_past_key_value = None, None

    self_attention_outputs = self.layer[0](
        hidden_states,
        attention_mask=attention_mask,
        position_bias=position_bias,
        layer_head_mask=layer_head_mask,
        past_key_value=self_attn_past_key_value,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    hidden_states, present_key_value_state = self_attention_outputs[:2]
    attention_outputs = self_attention_outputs[
        2:
    ]  # Keep self-attention outputs and relative position weights

    # clamp inf values to enable fp16 training
    if hidden_states.dtype == torch.float16:
        clamp_value = torch.where(
            torch.isinf(hidden_states).any(),
            torch.finfo(hidden_states.dtype).max - 1000,
            torch.finfo(hidden_states.dtype).max,
        )
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    do_cross_attention = self.is_decoder and encoder_hidden_states is not None
    if do_cross_attention:
        # the actual query length is unknown for cross attention
        # if using past key value states. Need to inject it here
        if present_key_value_state is not None:
            query_length = present_key_value_state[0].shape[2]
        else:
            query_length = None

        cross_attention_outputs = self.layer[1](
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            position_bias=encoder_decoder_position_bias,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = cross_attention_outputs[0]

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        # Combine self attn and cross attn key value states
        if present_key_value_state is not None:
            present_key_value_state = (
                present_key_value_state + cross_attention_outputs[1]
            )

        # Keep cross-attention outputs and relative position weights
        attention_outputs = attention_outputs + cross_attention_outputs[2:]

    # Apply Feed Forward layer
    if hasattr(self, "linear_gelu"):
        forwarded_states = self.layer[-1].layer_norm(hidden_states)
        hidden_gelu = self.linear_gelu(forwarded_states)
        forwarded_states = self.linear_mul(forwarded_states, hidden_gelu)
        if not self.distributed:
            if (
                hasattr(self.linear_add.linear, "weight")
                and isinstance(self.linear_add.linear.weight, torch.Tensor)
                and forwarded_states.dtype != self.linear_add.linear.weight.dtype
                and self.linear_add.linear.weight.dtype not in [torch.int8, torch.uint8]
            ):
                forwarded_states = forwarded_states.to(
                    self.linear_add.linear.weight.dtype
                )
            hidden_states = self.linear_add(forwarded_states, hidden_states)
        else:
            if (
                hasattr(self.layer[-1].DenseReluDense.wo, "weight")
                and isinstance(self.layer[-1].DenseReluDense.wo.weight, torch.Tensor)
                and forwarded_states.dtype
                != self.layer[-1].DenseReluDense.wo.weight.dtype
                and self.layer[-1].DenseReluDense.wo.weight.dtype
                not in [torch.int8, torch.uint8]
            ):
                forwarded_states = forwarded_states.to(
                    self.layer[-1].DenseReluDense.wo.weight.dtype
                )
            forwarded_states = self.layer[-1].DenseReluDense.wo(forwarded_states)
            hidden_states = hidden_states + self.layer[-1].DenseReluDense.dropout(
                forwarded_states
            )
    else:
        hidden_states = self.layer[-1](hidden_states)

    # clamp inf values to enable fp16 training
    if hidden_states.dtype == torch.float16:
        clamp_value = torch.where(
            torch.isinf(hidden_states).any(),
            torch.finfo(hidden_states.dtype).max - 1000,
            torch.finfo(hidden_states.dtype).max,
        )
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    outputs = (hidden_states,)

    if use_cache:
        outputs = outputs + (present_key_value_state,) + attention_outputs
    else:
        outputs = outputs + attention_outputs

    # hidden-states, present_key_value_states,
    # (self-attention position bias), (self-attention weights),
    # (cross-attention position bias), (cross-attention weights)
    return outputs


def MistralDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    if not self.distributed:
        hidden_states = self.mha_linear_add(hidden_states, residual)
    else:
        hidden_states = self.self_attn.o_proj(hidden_states)
        hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    mlp_gate = self.linear_silu_mul(hidden_states)

    if not self.distributed:
        hidden_states = self.mlp_linear_add(mlp_gate, residual)
    else:
        hidden_states = self.mlp.down_proj(mlp_gate)
        hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def MixtralDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    output_router_logits: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if "padding_mask" in kwargs:
        logger.warning(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`",
            _type=WarningType.DeprecatedArgument,
        )
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, sequence_length)` where padding elements are indicated by 0.
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        output_router_logits (`bool`, *optional*):
            Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
            should not be returned during inference.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
    """

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    if not self.distributed:
        hidden_states = self.mha_linear_add(hidden_states, residual)
    else:
        hidden_states = self.self_attn.o_proj(hidden_states)
        hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    # hidden_states, router_logits = self.block_sparse_moe(hidden_states)

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.block_sparse_moe.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, self.block_sparse_moe.top_k, dim=-1
    )
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=self.block_sparse_moe.num_experts
    ).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.block_sparse_moe.num_experts):
        expert_layer = self.block_sparse_moe.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        if expert_layer.w1.weight.dtype in [torch.qint8, torch.int8, torch.uint8]:
            final_hidden_states = torch.ops.torch_ipex.mixtral_moe_woq(
                hidden_states,
                top_x,
                idx,
                expert_layer.w1._op_context.get_data_handle(),
                expert_layer.w3._op_context.get_data_handle(),
                expert_layer.w2._op_context.get_data_handle(),
                routing_weights,
                final_hidden_states,
                self.distributed,
            )
        elif hasattr(expert_layer.w1, "use_dnnl") and expert_layer.w1.use_dnnl:
            final_hidden_states = torch.ops.torch_ipex.mixtral_moe(
                hidden_states,
                top_x,
                idx,
                expert_layer.w1._get_forward_weight(),
                expert_layer.w1.ctx.get_data_handle(),
                expert_layer.w3._get_forward_weight(),
                expert_layer.w3.ctx.get_data_handle(),
                expert_layer.w2._get_forward_weight(),
                expert_layer.w2.ctx.get_data_handle(),
                hasattr(expert_layer.w1, "use_dnnl") and expert_layer.w1.use_dnnl,
                routing_weights,
                final_hidden_states,
                self.distributed,
            )
        else:
            final_hidden_states = torch.ops.torch_ipex.mixtral_moe_tpp(
                hidden_states,
                top_x,
                idx,
                expert_layer.w1.weight,
                expert_layer.w3.weight,
                expert_layer.w2.weight,
                (
                    expert_layer.w1.tpp_fallback
                    if hasattr(expert_layer.w1, "tpp_fallback")
                    else True
                ),
                routing_weights,
                final_hidden_states,
                self.distributed,
            )
    final_hidden_states = final_hidden_states.reshape(
        batch_size, sequence_length, hidden_dim
    )
    hidden_states, router_logits = final_hidden_states, router_logits

    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    if output_router_logits:
        outputs += (router_logits,)

    return outputs


def MptBlock_forward(
    self,
    hidden_states: torch.Tensor,
    position_bias: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = True,
    output_attentions: bool = False,
):
    # hidden_states: [batch_size, seq_length, hidden_size]
    # Layer norm at the beginning of the transformer layer.
    layernorm_output = self.norm_1(hidden_states)

    residual = hidden_states

    # Self attention.
    attn_outputs, attn_weights, past_key_value = self.attn(
        layernorm_output,
        position_bias=position_bias,
        attention_mask=attention_mask,
        past_key_value=layer_past,
    )

    hidden_states = self.resid_attn_dropout(attn_outputs) + residual

    layernorm_output = self.norm_2(hidden_states)

    # Get residual
    residual = hidden_states

    # MLP.
    output = self.linear_gelu(layernorm_output)
    if not self.distributed:
        output = self.linear_add(output, residual)
    else:
        output = self.ffn.down_proj(output)
        output = output + residual
    outputs = (output,)

    if use_cache:
        outputs += (past_key_value,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # hidden_states, present, attentions


def StableLMEpochDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    self_attn_output, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    if hasattr(self, "use_parallel_residual") and self.use_parallel_residual:
        self_attn_output = self.self_attn.o_proj(self_attn_output)
        mlp_gate = self.linear_silu_mul(hidden_states)
        if not self.distributed:
            hidden_states = self.mlp_linear_add_add(
                mlp_gate, residual, self_attn_output
            )
        else:
            hidden_states = self.mlp.down_proj(mlp_gate)
            hidden_states = residual + self_attn_output + hidden_states
    else:
        if not self.distributed:
            hidden_states = self.mha_linear_add(self_attn_output, residual)
        else:
            hidden_states = self.self_attn.o_proj(self_attn_output)
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_gate = self.linear_silu_mul(hidden_states)

        if not self.distributed:
            hidden_states = self.mlp_linear_add(mlp_gate, residual)
        else:
            hidden_states = self.mlp.down_proj(mlp_gate)
            hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def QWenBlock_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    rotary_pos_emb_list: Optional[List[List[torch.Tensor]]] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    layernorm_output = self.ln_1(hidden_states)

    attn_outputs = self.attn(
        hidden_states=layernorm_output,
        rotary_pos_emb=rotary_pos_emb_list,
        layer_past=layer_past,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    attn_output = attn_outputs[0]

    outputs = attn_outputs[1:]

    residual = hidden_states
    if not self.distributed:
        layernorm_input = self.mha_linear_add(attn_output, residual)
    else:
        attn_output = self.attn.c_proj(attn_output)
        layernorm_input = attn_output + residual

    layernorm_output = self.ln_2(layernorm_input)

    residual = layernorm_input
    # mlp_output = self.mlp(layernorm_output)

    mlp_gate = self.linear_silu_mul(layernorm_output)

    if not self.distributed:
        hidden_states = self.mlp_linear_add(mlp_gate, residual)
    else:
        hidden_states = self.mlp.c_proj(mlp_gate)
        hidden_states = residual + hidden_states

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs


def Qwen2DecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    if not self.distributed:
        hidden_states = self.mha_linear_add(hidden_states, residual)
    else:
        hidden_states = self.self_attn.o_proj(hidden_states)
        hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    mlp_gate = self.linear_silu_mul(hidden_states)

    if not self.distributed:
        hidden_states = self.mlp_linear_add(mlp_gate, residual)
    else:
        hidden_states = self.mlp.down_proj(mlp_gate)
        hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def GitLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    output_attentions: Optional[bool] = False,
    pixel_values_present: Optional[bool] = False,
) -> Tuple[torch.Tensor]:
    self_attn_past_key_value = None
    if past_key_value is not None:
        if len(past_key_value) == 4:
            self_attn_past_key_value = past_key_value
        else:
            self_attn_past_key_value = past_key_value[:2]
    self_attention_outputs = self.attention.self(
        hidden_states,
        attention_mask=attention_mask,
        head_mask=head_mask,
        output_attentions=output_attentions,
        past_key_value=self_attn_past_key_value,
        pixel_values_present=pixel_values_present,
    )
    if not self.distributed:
        attention_output = self.mha_linear_add(self_attention_outputs[0], hidden_states)
    else:
        attention_output = self.attention.output.dense(self_attention_outputs[0])
        attention_output = attention_output + hidden_states
    attention_output = self.attention.output.LayerNorm(attention_output)

    # if decoder, the last output is tuple of self-attn cache
    outputs = self_attention_outputs[1:-1]
    present_key_value = self_attention_outputs[-1]
    intermediate_output = self.linear_gelu(attention_output)
    if not self.distributed:
        layer_output = self.mlp_linear_add(intermediate_output, attention_output)
    else:
        layer_output = self.output.dense(intermediate_output)
        layer_output = layer_output + attention_output
    layer_output = self.output.LayerNorm(layer_output)
    outputs = (layer_output,) + outputs

    # if decoder, return the attn key/values as the last output
    outputs = outputs + (present_key_value,)

    return outputs


def GitVisionEncoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.FloatTensor]:
    residual = hidden_states

    hidden_states = self.layer_norm1(hidden_states)
    hidden_states, attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        vision=True,
    )
    if not self.distributed:
        hidden_states = self.vision_mha_linear_add(hidden_states, residual)
    else:
        hidden_states = self.self_attn.out_proj(hidden_states)
        hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    # hidden_states = self.vision_linear_gelu(hidden_states)
    hidden_states = self.mlp.fc1(hidden_states)
    hidden_states = self.mlp.activation_fn(hidden_states)
    if not self.distributed:
        hidden_states = self.vision_mlp_linear_add(hidden_states, residual)
    else:
        hidden_states = self.mlp.fc2(hidden_states)
        hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def CLIPEncoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_attention_mask: torch.Tensor,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.FloatTensor]:
    residual = hidden_states

    hidden_states = self.layer_norm1(hidden_states)
    hidden_states, attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        encoder_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        vision=True,
    )
    if not self.distributed:
        hidden_states = self.vision_mha_linear_add(hidden_states, residual)
    else:
        hidden_states = self.self_attn.out_proj(hidden_states)
        hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    # hidden_states = self.mlp(hidden_states)
    hidden_states = self.mlp.fc1(hidden_states)
    hidden_states = self.mlp.activation_fn(hidden_states)
    if not self.distributed:
        hidden_states = self.vision_mlp_linear_add(hidden_states, residual)
    else:
        hidden_states = self.mlp.fc2(hidden_states)
        hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def YuanDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    if not self.distributed:
        hidden_states = self.mha_linear_add(hidden_states, residual)
    else:
        hidden_states = self.self_attn.o_proj(hidden_states)
        hidden_states = residual + hidden_states
    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    # hidden_states = self.mlp(hidden_states)
    # hidden_states = residual + hidden_states
    mlp_gate = self.linear_silu_mul(hidden_states)
    if not self.distributed:
        hidden_states = self.mlp_linear_add(mlp_gate, residual)
    else:
        hidden_states = self.mlp.down_proj(mlp_gate)
        hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    outputs += (present_key_value,)

    return outputs


def PhiDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    attn_outputs, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    # feed_forward_hidden_states = self.mlp(hidden_states)
    feed_forward_hidden_states = self.linear_gelu(hidden_states)
    if not self.distributed:
        hidden_states = self.linear_add_add(
            feed_forward_hidden_states, attn_outputs, residual
        )
    else:
        feed_forward_hidden_states = self.mlp.fc2(feed_forward_hidden_states)
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)
    return outputs


def Phi3DecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    attn_outputs, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    if not self.distributed:
        hidden_states = self.mha_linear_add(attn_outputs, residual)
    else:
        attn_outputs = self.self_attn.o_proj(attn_outputs)
        hidden_states = residual + attn_outputs

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    # hidden_states = self.mlp(hidden_states)
    up_states = self.mlp.gate_up_proj(hidden_states)
    gate, up_states = up_states.chunk(2, dim=-1)
    up_states = up_states * self.mlp.activation_fn(gate)
    if not self.distributed:
        hidden_states = self.mlp_linear_add(up_states, residual)
    else:
        hidden_states = self.mlp.down_proj(up_states)
        hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def WhisperEncoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_head_mask: torch.Tensor,
    output_attentions: bool = False,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states, attn_weights, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
    )
    if not self.distributed:
        hidden_states = self.mha_linear_add(hidden_states, residual)
    else:
        hidden_states = self.self_attn.out_proj(hidden_states)
        hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.linear_gelu(hidden_states)
    if not self.distributed:
        hidden_states = self.mlp_linear_add(hidden_states, residual)
    else:
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

    if hidden_states.dtype == torch.float16 and (
        torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
    ):
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def WhisperDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = True,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    self_attn_past_key_value = (
        past_key_value[:4] if past_key_value is not None else None
    )
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=self_attn_past_key_value,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
    )
    if not self.distributed:
        hidden_states = self.mha_linear_add(hidden_states, residual)
    else:
        hidden_states = self.self_attn.out_proj(hidden_states)
        hidden_states = residual + hidden_states
    cross_attn_present_key_value = None
    cross_attn_weights = None
    if encoder_hidden_states is not None:
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        cross_attn_past_key_value = (
            past_key_value[4:] if past_key_value is not None else None
        )
        (
            hidden_states,
            cross_attn_weights,
            cross_attn_present_key_value,
        ) = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value,
            output_attentions=output_attentions,
        )
        if not self.distributed:
            hidden_states = self.encoder_mha_linear_add(hidden_states, residual)
        else:
            hidden_states = self.encoder_attn.out_proj(hidden_states)
            hidden_states = residual + hidden_states
        present_key_value = present_key_value + cross_attn_present_key_value

    # Fully Connected
    residual = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)

    hidden_states = self.linear_gelu(hidden_states)
    if not self.distributed:
        hidden_states = self.mlp_linear_add(hidden_states, residual)
    else:
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights, cross_attn_weights)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def MllamaCrossAttentionDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    cross_attention_states: torch.Tensor,
    cross_attention_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, attn_weights, past_key_value = self.cross_attn(
        hidden_states=hidden_states,
        attention_mask=cross_attention_mask,
        cross_attention_states=cross_attention_states,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
    )
    hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    mlp_gate = self.linear_silu_mul(hidden_states)
    hidden_states = self.mlp.down_proj(mlp_gate)

    if full_text_row_masked_out_mask is not None:
        hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore
    hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    if use_cache:
        outputs += (past_key_value,)

    return outputs


def GPTNeoXLayer_forward(
    self,
    hidden_states: Optional[torch.FloatTensor],
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
):
    attention_layer_outputs = self.attention(
        self.input_layernorm(hidden_states),
        attention_mask=attention_mask,
        position_ids=position_ids,
        layer_past=layer_past,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    attn_output = attention_layer_outputs[
        0
    ]  # output_attn: attn_output, present, (attn_weights)
    attn_output = self.post_attention_dropout(attn_output)
    outputs = attention_layer_outputs[1:]

    if self.use_parallel_residual:
        # pseudocode:
        # x = x + attn(ln1(x)) + mlp(ln2(x))
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        mlp_output = self.post_mlp_dropout(mlp_output)
        hidden_states = mlp_output + attn_output + hidden_states
    else:
        # pseudocode:
        # x = x + attn(ln1(x))
        # x = x + mlp(ln2(x))
        attn_output = attn_output + hidden_states
        mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
        mlp_output = self.post_mlp_dropout(mlp_output)
        hidden_states = mlp_output + attn_output

    if use_cache:
        outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
    else:
        outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

    return outputs


class _IPEXDecoderLayerRef(nn.Module):
    def __init__(self, module, config, distributed=False):
        super().__init__()
        for k, v in module.__dict__.items():
            setattr(self, k, v)
        for k, v in module.__class__.__dict__.items():
            if k.startswith("__") or k.startswith("forward"):
                continue
            setattr(self.__class__, k, getattr(module.__class__, k))
        self.distributed = distributed
        self.model_backbone = config.architectures[0]
        if self.model_backbone in ["GPTJForCausalLM", "CodeGenForCausalLM"]:
            if not self.distributed:
                self.linear_add_add = _IPEXlinearAddAddRef(module.mlp.fc_out)
                del self.__dict__["_modules"]["mlp"].fc_out
            self.linear_gelu = _IPEXlinearNewGeluRef(module.mlp.fc_in)
            del self.__dict__["_modules"]["mlp"].fc_in
        elif self.model_backbone in [
            "LlamaForCausalLM",
            "MllamaForConditionalGeneration",
            "BaichuanForCausalLM",
            "MistralForCausalLM",
            "Qwen2ForCausalLM",
        ]:
            self.is_cross_decoder = False
            if (
                self.model_backbone == "MllamaForConditionalGeneration"
                and module._get_name() == "MllamaCrossAttentionDecoderLayer"
            ):
                self.is_cross_decoder = True
            else:
                if not self.distributed:
                    self.mha_linear_add = _IPEXlinearAddRef(module.self_attn.o_proj)
                    self.mlp_linear_add = _IPEXlinearAddRef(module.mlp.down_proj)
                    del self.__dict__["_modules"]["self_attn"].o_proj
                    del self.__dict__["_modules"]["mlp"].down_proj
            self.linear_silu_mul = _IPEXlinearSiluMulRef(
                module.mlp.gate_proj, module.mlp.up_proj
            )
            del self.__dict__["_modules"]["mlp"].gate_proj
            del self.__dict__["_modules"]["mlp"].up_proj
        elif self.model_backbone == "StableLmForCausalLM":
            if not self.distributed:
                if (
                    hasattr(self, "use_parallel_residual")
                    and self.use_parallel_residual
                ):
                    self.mlp_linear_add_add = _IPEXlinearAddAddRef(module.mlp.down_proj)
                    del self.__dict__["_modules"]["mlp"].down_proj
                else:
                    self.mha_linear_add = _IPEXlinearAddRef(module.self_attn.o_proj)
                    self.mlp_linear_add = _IPEXlinearAddRef(module.mlp.down_proj)
                    del self.__dict__["_modules"]["self_attn"].o_proj
                    del self.__dict__["_modules"]["mlp"].down_proj
            self.linear_silu_mul = _IPEXlinearSiluMulRef(
                module.mlp.gate_proj, module.mlp.up_proj
            )
            del self.__dict__["_modules"]["mlp"].gate_proj
            del self.__dict__["_modules"]["mlp"].up_proj
        elif self.model_backbone == "OPTForCausalLM":
            if not self.distributed:
                self.mha_linear_add = _IPEXlinearAddRef(module.self_attn.out_proj)
                self.mlp_linear_add = _IPEXlinearAddRef(module.fc2)
                del self.__dict__["_modules"]["self_attn"].out_proj
                del self.__dict__["_modules"]["fc2"]
            self.linear_relu = _IPEXlinearReluRef(module.fc1)
            del self.__dict__["_modules"]["fc1"]
        elif (
            self.model_backbone == "FalconForCausalLM"
            or self.model_backbone == "RWForCausalLM"
        ):
            self.linear_gelu = _IPEXlinearGeluRef(module.mlp.dense_h_to_4h)
            del self.__dict__["_modules"]["mlp"].dense_h_to_4h
            if not self.distributed:
                if (
                    module.self_attention.new_decoder_architecture
                    or config.parallel_attn
                    or not hasattr(module, "input_layernorm")
                ):
                    self.linear_add_add = _IPEXlinearAddAddRef(self.mlp.dense_4h_to_h)
                else:
                    self.linear_add = _IPEXlinearAddRef(self.mlp.dense_4h_to_h)
                del self.__dict__["_modules"]["mlp"].dense_4h_to_h
        elif self.model_backbone == "BloomForCausalLM":
            self.linear_gelu = _IPEXlinearGeluRef(module.mlp.dense_h_to_4h)
            del self.__dict__["_modules"]["mlp"].dense_h_to_4h
            if not self.distributed:
                self.linear_add = _IPEXlinearAddRef(self.mlp.dense_4h_to_h)
                del self.__dict__["_modules"]["mlp"].dense_4h_to_h
        elif self.model_backbone == "ChatGLMModel":
            if not self.distributed:
                self.mha_linear_add = _IPEXlinearAddRef(module.self_attention.dense)
                self.mlp_linear_add = _IPEXlinearAddRef(module.mlp.dense_4h_to_h)
                del self.__dict__["_modules"]["self_attention"].dense
                del self.__dict__["_modules"]["mlp"].dense_4h_to_h

            if module.mlp.dense_h_to_4h.weight.dtype not in [torch.uint8]:
                gate_weights, up_weights = module.mlp.dense_h_to_4h.weight.chunk(
                    2, dim=0
                )
                has_bias = module.mlp.dense_h_to_4h.bias is not None
                gate_linear = torch.nn.Linear(
                    gate_weights.shape[1], gate_weights.shape[0], has_bias
                )
                up_linear = torch.nn.Linear(
                    up_weights.shape[1], up_weights.shape[0], has_bias
                )
                gate_linear.weight.data = gate_weights
                up_linear.weight.data = up_weights
                if has_bias:
                    gate_bias, up_bias = module.mlp.dense_h_to_4h.bias.chunk(2, dim=0)
                    gate_linear.bias.data = gate_bias
                    up_linear.bias.data = up_bias
                self.linear_silu_mul = _IPEXlinearSiluMulRef(gate_linear, up_linear)
                del self.__dict__["_modules"]["mlp"].dense_h_to_4h
        elif self.model_backbone == "GPTBigCodeForCausalLM":
            self.linear_gelu = _IPEXlinearGeluRef(module.mlp.c_fc)
            del self.__dict__["_modules"]["mlp"].c_fc
            if not self.distributed:
                self.mha_linear_add = _IPEXlinearAddRef(module.attn.c_proj)
                self.mlp_linear_add = _IPEXlinearAddRef(module.mlp.c_proj)
                del self.__dict__["_modules"]["attn"].c_proj
                del self.__dict__["_modules"]["mlp"].c_proj
        elif self.model_backbone == "T5ForConditionalGeneration":
            if config.feed_forward_proj == "gated-gelu":
                self.linear_gelu = _IPEXlinearGeluRef(
                    module.layer[-1].DenseReluDense.wi_0
                )
                self.linear_mul = _IPEXlinearMulRef(
                    module.layer[-1].DenseReluDense.wi_1
                )
                del self.__dict__["_modules"]["layer"][-1].DenseReluDense.wi_0
                del self.__dict__["_modules"]["layer"][-1].DenseReluDense.wi_1
                if not self.distributed:
                    self.linear_add = _IPEXlinearAddRef(
                        module.layer[-1].DenseReluDense.wo
                    )
                    del self.__dict__["_modules"]["layer"][-1].DenseReluDense.wo
        elif self.model_backbone == "MptForCausalLM":
            self.linear_gelu = _IPEXlinearGeluRef(module.ffn.up_proj)
            del self.__dict__["_modules"]["ffn"].up_proj
            if not self.distributed:
                self.linear_add = _IPEXlinearAddRef(module.ffn.down_proj)
                del self.__dict__["_modules"]["ffn"].down_proj
        elif self.model_backbone == "MixtralForCausalLM":
            if not self.distributed:
                self.mha_linear_add = _IPEXlinearAddRef(module.self_attn.o_proj)
                del self.__dict__["_modules"]["self_attn"].o_proj
        elif self.model_backbone == "QWenLMHeadModel":
            if not self.distributed:
                self.mha_linear_add = _IPEXlinearAddRef(module.attn.c_proj)
                del self.__dict__["_modules"]["attn"].c_proj
                self.mlp_linear_add = _IPEXlinearAddRef(module.mlp.c_proj)
                del self.__dict__["_modules"]["mlp"].c_proj
            self.linear_silu_mul = _IPEXlinearSiluMulRef(module.mlp.w2, module.mlp.w1)
            del self.__dict__["_modules"]["mlp"].w2
            del self.__dict__["_modules"]["mlp"].w1
        elif self.model_backbone == "GitForCausalLM":
            if not self.distributed:
                if hasattr(module, "attention"):
                    self.mha_linear_add = _IPEXlinearAddRef(
                        module.attention.output.dense
                    )
                    del self.__dict__["_modules"]["attention"].output.dense
                if hasattr(module, "output"):
                    self.mlp_linear_add = _IPEXlinearAddRef(module.output.dense)
                    del self.__dict__["_modules"]["output"].dense
                if hasattr(module, "self_attn"):
                    self.vision_mha_linear_add = _IPEXlinearAddRef(
                        module.self_attn.out_proj
                    )
                    del self.__dict__["_modules"]["self_attn"].out_proj
                if hasattr(module, "mlp"):
                    self.vision_mlp_linear_add = _IPEXlinearAddRef(module.mlp.fc2)
                    del self.__dict__["_modules"]["mlp"].fc2
            if hasattr(module, "intermediate"):
                self.linear_gelu = _IPEXlinearGeluRef(module.intermediate.dense)
                del self.__dict__["_modules"]["intermediate"].dense
            # if hasattr(module, "mlp"):
            #     self.vision_linear_gelu = _IPEXlinearGeluRef(module.mlp.fc1)
            #     del self.__dict__["_modules"]["mlp"].fc1
        elif self.model_backbone == "LlavaLlamaForCausalLM":
            if not self.distributed:
                if hasattr(module.self_attn, "o_proj"):
                    self.mha_linear_add = _IPEXlinearAddRef(module.self_attn.o_proj)
                    del self.__dict__["_modules"]["self_attn"].o_proj
                if hasattr(module.self_attn, "out_proj"):
                    self.vision_mha_linear_add = _IPEXlinearAddRef(
                        module.self_attn.out_proj
                    )
                    del self.__dict__["_modules"]["self_attn"].out_proj
                if hasattr(module.mlp, "down_proj"):
                    self.mlp_linear_add = _IPEXlinearAddRef(module.mlp.down_proj)
                    del self.__dict__["_modules"]["mlp"].down_proj
                if hasattr(module.mlp, "fc2"):
                    self.vision_mlp_linear_add = _IPEXlinearAddRef(module.mlp.fc2)
                    del self.__dict__["_modules"]["mlp"].fc2
            if hasattr(module.mlp, "gate_proj") and hasattr(module.mlp, "up_proj"):
                self.linear_silu_mul = _IPEXlinearSiluMulRef(
                    module.mlp.gate_proj, module.mlp.up_proj
                )
                del self.__dict__["_modules"]["mlp"].gate_proj
                del self.__dict__["_modules"]["mlp"].up_proj
        elif self.model_backbone == "YuanForCausalLM":
            if not self.distributed:
                self.mha_linear_add = _IPEXlinearAddRef(module.self_attn.o_proj)
                self.mlp_linear_add = _IPEXlinearAddRef(module.mlp.down_proj)
                del self.__dict__["_modules"]["self_attn"].o_proj
                del self.__dict__["_modules"]["mlp"].down_proj
            self.linear_silu_mul = _IPEXlinearSiluMulRef(
                module.mlp.up_proj, module.mlp.gate_proj
            )
            del self.__dict__["_modules"]["mlp"].gate_proj
            del self.__dict__["_modules"]["mlp"].up_proj
        elif self.model_backbone == "PhiForCausalLM":
            if not self.distributed:
                self.linear_add_add = _IPEXlinearAddAddRef(module.mlp.fc2)
                del self.__dict__["_modules"]["mlp"].fc2
            self.linear_gelu = _IPEXlinearNewGeluRef(module.mlp.fc1)
            del self.__dict__["_modules"]["mlp"].fc1
        elif self.model_backbone == "Phi3ForCausalLM":
            if not self.distributed:
                self.mlp_linear_add = _IPEXlinearAddRef(module.mlp.down_proj)
                del self.__dict__["_modules"]["mlp"].down_proj
                self.mha_linear_add = _IPEXlinearAddRef(module.self_attn.o_proj)
                del self.__dict__["_modules"]["self_attn"].o_proj
        elif self.model_backbone == "WhisperForConditionalGeneration":
            if not self.distributed:
                self.mha_linear_add = _IPEXlinearAddRef(module.self_attn.out_proj)
                del self.__dict__["_modules"]["self_attn"].out_proj
                self.mlp_linear_add = _IPEXlinearAddRef(module.fc2)
                del self.__dict__["_modules"]["fc2"]
                if hasattr(module, "encoder_attn"):
                    self.encoder_mha_linear_add = _IPEXlinearAddRef(
                        module.encoder_attn.out_proj
                    )
                    del self.__dict__["_modules"]["encoder_attn"].out_proj
            self.linear_gelu = _IPEXlinearGeluRef(module.fc1)
            del self.__dict__["_modules"]["fc1"]
        else:
            AssertionError(False, "Do not support the optimization of your model yet")

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        cross_attention_states: torch.Tensor = None,
        cross_attention_mask: torch.Tensor = None,
        full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        alibi: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        encoder_decoder_position_bias: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        output_router_logits: Optional[bool] = False,
        pixel_values_present: Optional[bool] = False,
        vision: Optional[bool] = False,
    ):
        if self.model_backbone in ["GPTJForCausalLM", "CodeGenForCausalLM"]:
            return GPTJBlock_forward(
                self,
                hidden_states,
                layer_past,
                attention_mask,
                position_ids,
                head_mask,
                use_cache,
                output_attentions,
            )
        elif self.model_backbone == "GPTNeoXForCausalLM":
            return GPTNeoXLayer_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                head_mask,
                use_cache,
                layer_past,
                output_attentions,
            )
        elif (
            self.model_backbone == "LlamaForCausalLM"
            or self.model_backbone == "MllamaForConditionalGeneration"
        ):
            if not self.is_cross_decoder:
                return LlamaDecoderLayer_forward(
                    self,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                return MllamaCrossAttentionDecoderLayer_forward(
                    self,
                    hidden_states,
                    cross_attention_states,
                    cross_attention_mask,
                    attention_mask,
                    full_text_row_masked_out_mask,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
        elif self.model_backbone == "Qwen2ForCausalLM":
            return Qwen2DecoderLayer_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        elif self.model_backbone == "OPTForCausalLM":
            return OPTDecoderLayer_forward(
                self,
                hidden_states,
                attention_mask,
                layer_head_mask,
                output_attentions,
                use_cache,
                past_key_value,
            )
        elif (
            self.model_backbone == "FalconForCausalLM"
            or self.model_backbone == "RWForCausalLM"
        ):
            return FalconDecoderLayer_forward(
                self,
                hidden_states,
                alibi,
                attention_mask,
                layer_past,
                head_mask,
                use_cache,
                output_attentions,
            )
        elif self.model_backbone == "BloomForCausalLM":
            return BloomBlock_forward(
                self,
                hidden_states,
                alibi,
                attention_mask,
                layer_past,
                head_mask,
                use_cache,
                output_attentions,
            )
        elif self.model_backbone == "BaichuanForCausalLM":
            return BaichuanDecoder_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        elif self.model_backbone == "ChatGLMModel":
            return GLMBlock_forward(
                self, hidden_states, attention_mask, rotary_pos_emb, kv_cache, use_cache
            )
        elif self.model_backbone == "GPTBigCodeForCausalLM":
            return GPTBigCodeBlock_forward(
                self,
                hidden_states,
                layer_past,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache,
                output_attentions,
            )
        elif self.model_backbone == "T5ForConditionalGeneration":
            return T5Block_forward(
                self,
                hidden_states,
                attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_decoder_position_bias,
                layer_head_mask,
                cross_attn_layer_head_mask,
                past_key_value,
                use_cache,
                output_attentions,
            )
        elif self.model_backbone == "MistralForCausalLM":
            return MistralDecoderLayer_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        elif self.model_backbone == "MptForCausalLM":
            return MptBlock_forward(
                self,
                hidden_states,
                position_bias,
                attention_mask,
                layer_past,
                use_cache,
                output_attentions,
            )
        elif self.model_backbone == "MixtralForCausalLM":
            return MixtralDecoderLayer_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                output_router_logits,
                use_cache,
            )
        elif self.model_backbone == "StableLmForCausalLM":
            return StableLMEpochDecoderLayer_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        elif self.model_backbone == "QWenLMHeadModel":
            return QWenBlock_forward(
                self,
                hidden_states,
                rotary_pos_emb,
                layer_past,
                attention_mask,
                head_mask,
                use_cache,
                output_attentions,
            )
        elif self.model_backbone == "GitForCausalLM":
            if vision is not None and vision:
                return GitVisionEncoderLayer_forward(
                    self,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            return GitLayer_forward(
                self,
                hidden_states,
                attention_mask,
                head_mask,
                past_key_value,
                output_attentions,
                pixel_values_present,
            )
        elif self.model_backbone == "LlavaLlamaForCausalLM":
            if vision is not None and vision:
                return CLIPEncoderLayer_forward(
                    self,
                    hidden_states,
                    attention_mask,
                    encoder_attention_mask,
                    output_attentions,
                )
            return LlamaDecoderLayer_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        elif self.model_backbone == "YuanForCausalLM":
            return YuanDecoderLayer_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        elif self.model_backbone == "PhiForCausalLM":
            return PhiDecoderLayer_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                output_attentions,
                use_cache,
                past_key_value,
            )
        elif self.model_backbone == "Phi3ForCausalLM":
            return Phi3DecoderLayer_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                output_attentions,
                use_cache,
                past_key_value,
            )
        elif self.model_backbone == "WhisperForConditionalGeneration":
            if encoder_hidden_states is not None:
                return WhisperDecoderLayer_forward(
                    self,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            return WhisperEncoderLayer_forward(
                self, hidden_states, attention_mask, layer_head_mask, output_attentions
            )
        else:
            AssertionError(False, "Do not support the optimization of your model yet")
