import torch
from torch import nn
from typing import Optional, Tuple, Union


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


def LlamaMLP_forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


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
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def GPTJMLP_forward(
    self, hidden_states: Optional[torch.FloatTensor]
) -> torch.FloatTensor:
    hidden_states = torch.ops.torch_ipex.fc_in_gemm(
        hidden_states, self.fc_in.weight, self.fc_in.bias
    )
    return hidden_states


def GPTJMLP_forward_distributed(
    self, hidden_states: Optional[torch.FloatTensor]
) -> torch.FloatTensor:
    hidden_states = torch.ops.torch_ipex.fc_in_gemm(
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
    hidden_states = torch.ops.torch_ipex.fc_out_gemm(
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


def Apply_TPP_optimization(model, dtype, distributed=False):
    def convert_forward(m, target_m, new_forward):
        for _, sub_m in m.named_children():
            if isinstance(sub_m, target_m):
                bound_method = new_forward.__get__(sub_m, sub_m.__class__)
                setattr(sub_m, "forward", bound_method)
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
                "tpp rope optimization requires transformers package and its version between {} and {}, fallback due to not meet".format(
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
                "tpp rope optimization requires the transformers with version: between {} and {} while now transformers== {}, fallback due to not meet".format(
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
