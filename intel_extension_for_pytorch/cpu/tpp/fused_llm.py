import torch
from torch import nn
from typing import Optional, Tuple, Union
from torch.nn import functional as F


def GPTNeoXMLP_forward(self, hidden_states):
    hidden_states = self.dense_h_to_4h(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.dense_4h_to_h(hidden_states)
    return hidden_states


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
    outputs = attention_layer_outputs[1:]

    if self.use_parallel_residual:
        # pseudocode:
        # x = x + attn(ln1(x)) + mlp(ln2(x))
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = mlp_output + attn_output + hidden_states
    else:
        # pseudocode:
        # x = x + attn(ln1(x))
        # x = x + mlp(ln2(x))
        attn_output = attn_output + hidden_states
        mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
        hidden_states = mlp_output + attn_output

    if use_cache:
        outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
    else:
        outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

    return outputs


def LlamaMLP_forward_distributed(self, x):
    gate = torch.ops.torch_ipex.tpp_linear_silu(
        x, self.gate_proj.weight, x.new_empty(0)
    )
    up = torch.ops.torch_ipex.tpp_linear_mul(
        x, gate, self.up_proj.weight, x.new_empty(0)
    )
    return self.down_proj(up)


def LlamaMLP_forward(self, x):
    gate = torch.ops.torch_ipex.tpp_linear_silu(
        x, self.gate_proj.weight, x.new_empty(0)
    )
    up = torch.ops.torch_ipex.tpp_linear_mul(
        x, gate, self.up_proj.weight, x.new_empty(0)
    )
    return up


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
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = torch.ops.torch_ipex.tpp_linear_add(
        self.mlp(hidden_states),
        residual,
        self.mlp.down_proj.weight,
        hidden_states.new_empty(0),
        1.0,
    )

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
    # hidden_states = torch.ops.torch_ipex.tpp_linear_add(
    #     hidden_states,
    #     residual,
    #     self.self_attn.out_proj.weight,
    #     self.self_attn.out_proj.bias,
    #     1.0,
    # )
    hidden_states = residual + hidden_states

    # 350m applies layer norm AFTER attention
    if not self.do_layer_norm_before:
        hidden_states = self.self_attn_layer_norm(hidden_states)

    # Fully Connected
    hidden_states_shape = hidden_states.shape
    # TPP only supports 3d inputs for now
    # hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual = hidden_states

    # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
    if self.do_layer_norm_before:
        hidden_states = self.final_layer_norm(hidden_states)

    hidden_states = torch.ops.torch_ipex.tpp_linear_relu(
        hidden_states, self.fc1.weight, self.fc1.bias
    )

    hidden_states = torch.ops.torch_ipex.tpp_linear_add(
        hidden_states,
        residual,
        self.fc2.weight,
        self.fc2.bias,
        1.0,
    ).view(hidden_states_shape)

    # 350m applies layer norm AFTER attention
    if not self.do_layer_norm_before:
        hidden_states = self.final_layer_norm(hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def OPTDecoderLayer_forward_distributed(
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
    # hidden_states = self.self_attn.out_proj(hidden_states)
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.dropout, training=self.training
    )
    hidden_states = residual + hidden_states

    # 350m applies layer norm AFTER attention
    if not self.do_layer_norm_before:
        hidden_states = self.self_attn_layer_norm(hidden_states)

    # Fully Connected
    hidden_states_shape = hidden_states.shape
    # TPP only supports 3d inputs for now
    # hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual = hidden_states

    # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
    if self.do_layer_norm_before:
        hidden_states = self.final_layer_norm(hidden_states)

    hidden_states = torch.ops.torch_ipex.tpp_linear_relu(
        hidden_states, self.fc1.weight, self.fc1.bias
    )

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


def GPTJMLP_forward(
    self, hidden_states: Optional[torch.FloatTensor]
) -> torch.FloatTensor:
    hidden_states = torch.ops.torch_ipex.tpp_linear_gelu(
        hidden_states, self.fc_in.weight, self.fc_in.bias
    )
    return hidden_states


def GPTJMLP_forward_distributed(
    self, hidden_states: Optional[torch.FloatTensor]
) -> torch.FloatTensor:
    hidden_states = torch.ops.torch_ipex.tpp_linear_gelu(
        hidden_states, self.fc_in.weight, self.fc_in.bias
    )
    hidden_states = self.fc_out(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states


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

    feed_forward_hidden_states = self.mlp(hidden_states)
    hidden_states = torch.ops.torch_ipex.tpp_linear_add_add(
        feed_forward_hidden_states,
        attn_output,
        residual,
        self.mlp.fc_out.weight,
        self.mlp.fc_out.bias,
        1.0,
    )

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions)


def FalconMLP_forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.dense_h_to_4h.bias is not None:
        x = torch.ops.torch_ipex.tpp_linear_gelu(
            x, self.dense_h_to_4h.weight, self.dense_h_to_4h.bias
        )
    else:
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
    return x


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
    if self.self_attention.new_decoder_architecture or not hasattr(
        self, "input_layernorm"
    ):
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
    outputs = attn_outputs[1:]
    # MLP.
    mlp_output = self.mlp(mlp_layernorm_out)
    if (
        self.self_attention.new_decoder_architecture
        or self.config.parallel_attn
        or not hasattr(self, "input_layernorm")
    ):
        if self.mlp.dense_4h_to_h.bias is not None:
            output = torch.ops.torch_ipex.tpp_linear_add_add(
                mlp_output,
                attention_output,
                residual,
                self.mlp.dense_4h_to_h.weight,
                self.mlp.dense_4h_to_h.bias,
                1.0,
            )
        else:
            mlp_output += attention_output
            output = mlp_output + residual
    else:
        if self.mlp.dense_4h_to_h.bias is not None:
            output = torch.ops.torch_ipex.tpp_linear_add(
                mlp_output,
                residual,
                self.mlp.dense_4h_to_h.weight,
                self.mlp.dense_4h_to_h.bias,
                1.0,
            )
        else:
            output = mlp_output + residual

    if use_cache:
        outputs = (output,) + outputs
    else:
        outputs = (output,) + outputs[1:]
    return outputs  # hidden_states, present, attentions


def FalconMLP_forward_distributed(self, x: torch.Tensor) -> torch.Tensor:
    if self.dense_h_to_4h.bias is not None:
        x = torch.ops.torch_ipex.tpp_linear_gelu(
            x, self.dense_h_to_4h.weight, self.dense_h_to_4h.bias
        )
    else:
        x = self.act(self.dense_h_to_4h(x))
    return x


def FalconDecoderLayer_forward_distributed(
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

    if self.self_attention.new_decoder_architecture or not hasattr(
        self, "input_layernorm"
    ):
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
    # attention_output = self.self_attention.dense(attention_output)
    if not (
        self.self_attention.new_decoder_architecture
        or not hasattr(self, "input_layernorm")
    ):
        if self.config.parallel_attn:
            mlp_layernorm_out = attention_layernorm_out
        else:
            residual = attention_output + residual
            mlp_layernorm_out = self.post_attention_layernorm(residual)

    outputs = attn_outputs[1:]

    # MLP.
    mlp_output = self.mlp(mlp_layernorm_out)
    mlp_output = self.mlp.dense_4h_to_h(mlp_output)

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


def BloomMLP_forward(
    self, hidden_states: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    hidden_states = torch.ops.torch_ipex.tpp_linear_gelu(
        hidden_states, self.dense_h_to_4h.weight, self.dense_h_to_4h.bias
    )
    if self.pretraining_tp > 1 and self.slow_but_exact:
        intermediate_output = torch.zeros_like(residual)
        slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
        for i in range(self.pretraining_tp):
            intermediate_output = intermediate_output + F.linear(
                hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
            )
        output = intermediate_output + residual
    else:
        output = torch.ops.torch_ipex.tpp_linear_add(
            hidden_states,
            residual,
            self.dense_4h_to_h.weight,
            self.dense_4h_to_h.bias,
            1.0,
        )

    return output


def BloomMLP_forward_distributed(
    self, hidden_states: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    hidden_states = torch.ops.torch_ipex.tpp_linear_gelu(
        hidden_states, self.dense_h_to_4h.weight, self.dense_h_to_4h.bias
    )
    if self.pretraining_tp > 1 and self.slow_but_exact:
        intermediate_output = torch.zeros_like(residual)
        slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
        for i in range(self.pretraining_tp):
            intermediate_output = intermediate_output + F.linear(
                hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
            )
    else:
        intermediate_output = self.dense_4h_to_h(hidden_states)
    output = intermediate_output + residual

    return output


def GLMMLP_forward(self, hidden_states):
    # [s, b, 4hp]
    intermediate_parallel = self.dense_h_to_4h(hidden_states)
    intermediate_parallel = self.activation_func(intermediate_parallel)
    # [s, b, h]
    # output = self.dense_4h_to_h(intermediate_parallel)
    # return output
    return intermediate_parallel


def GLMBlock_forward(
    self,
    hidden_states,
    attention_mask,
    rotary_pos_emb,
    kv_cache=None,
    use_cache=True,
):
    layernorm_output = self.input_layernorm(hidden_states)
    # Self attention.
    attention_output, kv_cache = self.self_attention(
        layernorm_output,
        attention_mask,
        rotary_pos_emb,
        kv_cache=kv_cache,
        use_cache=use_cache,
    )
    # Residual connection.
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = hidden_states
    layernorm_input = torch.nn.functional.dropout(
        attention_output, p=self.hidden_dropout, training=self.training
    )
    layernorm_input = residual + layernorm_input

    # Layer norm post the self attention.
    layernorm_output = self.post_attention_layernorm(layernorm_input)

    # MLP.
    mlp_output = self.mlp(layernorm_output)

    # Second residual connection.
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = layernorm_input

    # output = torch.nn.functional.dropout(
    #     mlp_output, p=self.hidden_dropout, training=self.training
    # )
    # output = residual + output
    if self.mlp.dense_4h_to_h.bias is not None:
        output = torch.ops.torch_ipex.tpp_linear_add(
            mlp_output,
            residual,
            self.mlp.dense_4h_to_h.weight,
            self.mlp.dense_4h_to_h.bias,
            1.0,
        )
    else:
        output = self.mlp.dense_4h_to_h(mlp_output)
        output = residual + output

    return output, kv_cache


def GPTBigCodeMLP_forward(
    self, hidden_states: Optional[Tuple[torch.Tensor]]
) -> torch.Tensor:
    hidden_states = torch.ops.torch_ipex.tpp_linear_gelu(
        hidden_states, self.c_fc.weight, self.c_fc.bias
    )
    # hidden_states = self.c_proj(hidden_states)
    # hidden_states = self.dropout(hidden_states)
    return hidden_states


def GPTBigCodeMLP_forward_distributed(
    self, hidden_states: Optional[Tuple[torch.Tensor]]
) -> torch.Tensor:
    hidden_states = torch.ops.torch_ipex.tpp_linear_gelu(
        hidden_states, self.c_fc.weight, self.c_fc.bias
    )
    hidden_states = self.c_proj(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states


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
    feed_forward_hidden_states = self.mlp(hidden_states)
    # residual connection
    # hidden_states = residual + feed_forward_hidden_states
    hidden_states = torch.ops.torch_ipex.tpp_linear_add(
        feed_forward_hidden_states,
        residual,
        self.mlp.c_proj.weight,
        self.mlp.c_proj.bias,
        1.0,
    )

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions, cross_attentions)


def GPTBigCodeBlock_forward_distributed(
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
    feed_forward_hidden_states = self.mlp(hidden_states)
    # residual connection
    hidden_states = residual + feed_forward_hidden_states

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions, cross_attentions)


def BaichuanMLP_forward(self, x):
    gate = torch.ops.torch_ipex.tpp_linear_silu(
        x, self.gate_proj.weight, x.new_empty(0)
    )
    up = torch.ops.torch_ipex.tpp_linear_mul(
        x, gate, self.up_proj.weight, x.new_empty(0)
    )
    return up


def BaichuanMLP_forward_distributed(self, x):
    gate = torch.ops.torch_ipex.tpp_linear_silu(
        x, self.gate_proj.weight, x.new_empty(0)
    )
    up = torch.ops.torch_ipex.tpp_linear_mul(
        x, gate, self.up_proj.weight, x.new_empty(0)
    )
    return self.down_proj(up)


def BaichuanLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
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
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = torch.ops.torch_ipex.tpp_linear_add(
        self.mlp(hidden_states),
        residual,
        self.mlp.down_proj.weight,
        hidden_states.new_empty(0),
        1.0,
    )

    outputs = (hidden_states,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def BaichuanLayer_forward_distributed(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
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
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def T5DenseGatedActDense_forward(self, hidden_states):
    # hidden_gelu = self.act(self.wi_0(hidden_states))
    hidden_gelu = torch.ops.torch_ipex.tpp_linear_gelu(
        hidden_states,
        self.wi_0.weight,
        torch.zeros(self.wi_0.out_features, dtype=hidden_states.dtype),
    )
    # hidden_linear = self.wi_1(hidden_states)
    # hidden_states = hidden_gelu * hidden_linear
    # hidden_states = self.dropout(hidden_states)
    hidden_states = torch.ops.torch_ipex.tpp_linear_mul(
        hidden_states,
        hidden_gelu,
        self.wi_1.weight,
        torch.zeros(self.wi_1.out_features, dtype=hidden_states.dtype),
    )

    if (
        isinstance(self.wo.weight, torch.Tensor)
        and hidden_states.dtype != self.wo.weight.dtype
    ):
        hidden_states = hidden_states.to(self.wo.weight.dtype)

    # hidden_states = self.wo(hidden_states)
    return hidden_states


def T5LayerFF_forward(self, hidden_states):
    forwarded_states = self.layer_norm(hidden_states)
    forwarded_states = self.DenseReluDense(forwarded_states)
    # hidden_states = hidden_states + self.dropout(forwarded_states)
    hidden_states = torch.ops.torch_ipex.tpp_linear_add(
        forwarded_states,
        hidden_states,
        self.DenseReluDense.wo.weight,
        torch.zeros(self.DenseReluDense.wo.out_features, dtype=hidden_states.dtype),
        1.0,
    )
    return hidden_states


def Apply_TPP_optimization(model, dtype, distributed=False):
    def convert_forward(m, target_m, new_forward):
        for _, sub_m in m.named_children():
            if isinstance(sub_m, target_m):
                bound_method = new_forward.__get__(sub_m, sub_m.__class__)
                sub_m.forward = bound_method
            convert_forward(sub_m, target_m, new_forward)

    import warnings

    warnings.warn(
        "Apply_TPP_optimization is a temp API, will be removed soon, please check the NEW ipex.optimze_transformers API"
    )
    from intel_extension_for_pytorch.cpu._auto_kernel_selection import _enable_tpp

    _enable_tpp()
    try:
        # tpp rope optimization has transformers version requirements
        import pkg_resources

        installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
        min_version = "4.28.0"
        max_version = "4.30.0"
        if "transformers" not in installed_pkg:
            raise RuntimeError(
                "tpp rope optimization requires transformers package and its version between {} and {}, \
                    fallback due to not meet".format(
                    min_version, max_version
                )
            )

        import transformers
        from packaging import version

        trans_version = transformers.__version__
        if version.parse(trans_version) < version.parse(min_version) or version.parse(
            trans_version
        ) > version.parse(max_version):
            raise RuntimeError(
                "tpp rope optimization requires the transformers with version: between {} and {} while now transformers== {},\
                     fallback due to not meet".format(
                    min_version, max_version, trans_version
                )
            )

        if not distributed:
            convert_forward(
                model,
                transformers.models.gptj.modeling_gptj.GPTJBlock,
                GPTJBlock_forward,
            )
            convert_forward(
                model, transformers.models.gptj.modeling_gptj.GPTJMLP, GPTJMLP_forward
            )
        else:
            convert_forward(
                model,
                transformers.models.gptj.modeling_gptj.GPTJMLP,
                GPTJMLP_forward_distributed,
            )

    except RuntimeError:
        pass
