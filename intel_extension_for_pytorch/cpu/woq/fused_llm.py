import torch
from typing import Optional, Tuple, Union


def GPTJMLP_woq_forward(
    self, hidden_states: Optional[torch.FloatTensor]
) -> torch.FloatTensor:
    if hasattr(self.fc_in, "_op_context") and self.fc_in._op_context is not None:
        hidden_states = torch.ops.torch_ipex.woq_linear_gelu(
            hidden_states, self.fc_in._op_context.get_data_handle()
        )
    else:
        hidden_states = self.fc_in(hidden_states)
    return hidden_states


def GPTJBlock_woq_forward(
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
    others = [attn_output, residual]
    if (
        hasattr(self.mlp.fc_out, "_op_context")
        and self.mlp.fc_out._op_context is not None
    ):
        hidden_states = torch.ops.torch_ipex.woq_linear_add_add(
            feed_forward_hidden_states,
            self.mlp.fc_out._op_context.get_data_handle(),
            others,
        )
    else:
        hidden_states = self.mlp.fc_out(feed_forward_hidden_states)

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions)
