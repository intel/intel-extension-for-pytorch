import torch
from torch import nn
from typing import Optional, Tuple, Union
import re
from ...reference.fusions.linear_fusion import (
    _IPEXlinearSiluRef,
    _IPEXlinearAddRef,
    _IPEXlinearAddAddRef,
    _IPEXlinearMulRef,
    _IPEXlinearNewGeluRef,
    _IPEXlinearReluRef,
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

    mlp_gate = self.linear_mul(hidden_states, self.linear_silu(hidden_states))

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

        if re.search("GPTJ", self.model_backbone, re.IGNORECASE):
            if not self.distributed:
                self.linear_add_add = _IPEXlinearAddAddRef(module.mlp.fc_out)
                del self.__dict__["_modules"]["mlp"].fc_out
            self.linear_gelu = _IPEXlinearNewGeluRef(module.mlp.fc_in)
            del self.__dict__["_modules"]["mlp"].fc_in
        elif re.search("llama", self.model_backbone, re.IGNORECASE):
            if not self.distributed:
                self.mha_linear_add = _IPEXlinearAddRef(module.self_attn.o_proj)
                self.mlp_linear_add = _IPEXlinearAddRef(module.mlp.down_proj)
                del self.__dict__["_modules"]["self_attn"].o_proj
                del self.__dict__["_modules"]["mlp"].down_proj
            self.linear_silu = _IPEXlinearSiluRef(module.mlp.gate_proj)
            self.linear_mul = _IPEXlinearMulRef(module.mlp.up_proj)
            del self.__dict__["_modules"]["mlp"].gate_proj
            del self.__dict__["_modules"]["mlp"].up_proj

        elif re.search("OPT", self.model_backbone, re.IGNORECASE):
            if not self.distributed:
                self.mha_linear_add = _IPEXlinearAddRef(module.self_attn.out_proj)
                self.mlp_linear_add = _IPEXlinearAddRef(module.fc2)
                del self.__dict__["_modules"]["self_attn"].out_proj
                del self.__dict__["_modules"]["fc2"]
            self.linear_relu = _IPEXlinearReluRef(module.fc1)
            del self.__dict__["_modules"]["fc1"]
        else:
            AssertionError(False, "Do not support the optimization of your model yet")

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):
        if re.search("GPTJ", self.model_backbone, re.IGNORECASE):
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
        elif re.search("llama", self.model_backbone, re.IGNORECASE):
            return LlamaDecoderLayer_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        elif re.search("OPT", self.model_backbone, re.IGNORECASE):
            return OPTDecoderLayer_forward(
                self,
                hidden_states,
                attention_mask,
                layer_head_mask,
                output_attentions,
                use_cache,
                past_key_value,
            )
        else:
            AssertionError(False, "Do not support the optimization of your model yet")
