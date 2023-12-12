import torch
from torch import nn
from typing import Optional, Tuple, Union
from ...reference.fusions.linear_fusion import (
    _IPEXlinearAddRef,
    _IPEXlinearAddAddRef,
    _IPEXlinearNewGeluRef,
    _IPEXlinearReluRef,
    _IPEXlinearGeluRef,
    _IPEXlinearSiluMulRef,
)


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

    # MLP.
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
        elif self.model_backbone in ["LlamaForCausalLM", "BaichuanForCausalLM"]:
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
        else:
            AssertionError(False, "Do not support the optimization of your model yet")

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
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
        elif self.model_backbone == "LlamaForCausalLM":
            return LlamaDecoderLayer_forward(
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
        else:
            AssertionError(False, "Do not support the optimization of your model yet")
