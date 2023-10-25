import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union

from .transformer_modules.RoPE import GPTJRotaryEmbedding
from ._transformers import MAX_OUT_SEQ_LEN
from .transformer_modules.QuantizedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16,
    IPEXTransformerAttnOptimizedInt4,
)  # noqa
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.Mlp import *  # noqa
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from transformers.modeling_outputs import CausalLMOutputWithPast
from .transformer_modules.Decoderblock import IPEXTransformerBlock
import sys
import os

acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in [
    "1",
    "ON",
    "Y",
    "YES",
    "TRUE",
]


class NewIPEXGPTJBlock(IPEXTransformerBlock):
    def __init__(
        self,
        module,
        config,
        dtype="fp16",
        device="xpu",
        module_name="",
        tp_size=1,
        tp_group=None,
    ):
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(
            config, device, dtype, tp_size, tp_group
        )
        self.attn = self.build_attention_from_config()
        self.mlp = self.build_mlp_from_config()
        self.ln = nn.LayerNorm(
            self.ipex_config.embedding_dim, eps=self.ipex_config.norm_eps
        )
        self.port_all_parameters_to_new_module()
        # self.mlp = IPEXGPTJMLP(config)

    def build_attention_from_config(self):
        dtype = self.ipex_config.dtype
        impl = self.ipex_config.impl
        attn_type = IPEXTransformerAttn
        attn_type_str = "IPEXTransformerAttn"
        for elem in [impl.name, dtype]:
            attn_type_str = attn_type_str + elem.capitalize()
            if hasattr(sys.modules[__name__], attn_type_str):
                attn_type = getattr(sys.modules[__name__], attn_type_str)
        return attn_type(self.ipex_config)

    def build_mlp_from_config(self):
        dtype = self.ipex_config.dtype
        impl = self.ipex_config.impl
        activation = self.ipex_config.ipex_act
        mlp_type = IPEXTransformerMLP
        mlp_type_str = "IPEXTransformerMLP"
        for elem in [impl.name, dtype, activation.name, "gptj"]:
            mlp_type_str = mlp_type_str + elem.capitalize()
            if hasattr(sys.modules[__name__], mlp_type_str):
                mlp_type = getattr(sys.modules[__name__], mlp_type_str)
        return mlp_type(self.ipex_config)

    def build_ipex_transformer_config(
        self, config, device, dtype, tp_size, tp_group
    ) -> IPEXTransformerConfig:
        activation_function = self.config.activation_function
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
            embedding_dim=self.config.n_embd,
            intermediate_dim=self.config.n_inner,
            num_attention_head=self.config.n_head,
            max_positions=self.config.n_positions,
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=GPTJRotaryEmbedding,
            rotary_dim=self.config.rotary_dim,
            use_casual_mask=True,
            activation_function=activation_function,
            ipex_act=ipex_activation,
            norm_eps=self.config.layer_norm_epsilon,
            residual_dropout=self.config.resid_pdrop,
            attn_dropout=self.config.attn_pdrop,
            residual_pdrop=self.config.resid_pdrop,
            scale_attention=True,
            dtype=dtype,
            tp_size=tp_size,
            tp_group=tp_group,
        )

    def port_attn_parameter(self):
        self.attn.load_parameter(
            self.module.attn.q_proj,
            self.module.attn.k_proj,
            self.module.attn.v_proj,
            self.module.attn.out_proj,
        )

    def port_mlp_parameter(self):
        self.mlp.load_parameter(self.module.mlp.fc_in, self.module.mlp.fc_out)

    def port_norm_parameter(self):
        self.ln.weight = self.module.ln_1.weight
        self.ln.bias = self.module.ln_1.bias

    def transpose_parameter(self):
        self.attn.transpose_parameter()
        self.mlp.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()
        self.attn.cat_qkv()

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        # hidden_states:  [bs*beam, seq, hidden_size]
        # position_ids:   [bs*beam, seq]
        # attention_mask: [bs*beam, head, q_seq, kv_seq]
        bs = IPEXTransformerAttn.batch_size
        dim = hidden_states.dim()
        if dim == 3:
            beam = hidden_states.shape[0] // bs
            seq = hidden_states.shape[1]
        elif dim == 4:
            beam = hidden_states.shape[1]
            seq = hidden_states.shape[2]
        else:
            print("Unsupported input shape")
            return
        IPEXTransformerAttn.beam_size = beam
        first_token = True if seq > 1 else False
        hidden_size = hidden_states.shape[-1]
        hidden_shape = [bs, beam, seq, hidden_size]
        if first_token and beam > 1:
            # for 1st token, keep the original layout
            # reduce the duplicated info in beam dim
            # shape -> [bs*beam, seq, hidden_size]
            # layout -> [bs*beam, seq, hidden_size]
            hidden_states = hidden_states.view(hidden_shape)[:, 0, :, :].contiguous()
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
        else:
            # for 2nd to last token, we convert the layout
            # shape -> [bs*beam, seq, hidden_size]
            # convert layout form [bs*beam, seq, hidden_size] to [seq, bs*beam, hidden_size]
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        residual = hidden_states
        hidden_states = torch.ops.torch_ipex.fast_layer_norm(
            hidden_states,
            self.ln.normalized_shape,
            self.ln.weight,
            self.ln.bias,
            self.ln.eps,
        )
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            first_token=first_token,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        hidden_states = self.mlp(hidden_states, attn_output, residual)
        if first_token and beam > 1:
            # for 1st token, expand the result with beam
            hidden_states = hidden_states.view(bs, 1, seq, hidden_size)
            hidden_states = hidden_states.expand([bs, beam, seq, hidden_size])
        else:
            # for 2nd to last token, we convert the layout back
            # convert hidden_states form [seq, beam, hidden_size] back to [beam, seq, hidden_size]
            hidden_states = hidden_states.transpose(0, 1)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs


def IPEXGPTJForCausalLMForward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
    """
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]
    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.transformer.first_device)
        hidden_states = hidden_states.to(self.lm_head.weight.device)

    # make sure sampling in fp16 works correctly and
    # compute loss in fp32 to match with mesh-tf version
    # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179

    if hidden_states.dim() > 3:
        hidden_states = hidden_states.reshape(
            [-1, hidden_states.shape[-2], hidden_states.shape[-1]]
        )
    if not acc_test:
        shape = list(hidden_states.size())
        shape[1] = 1
        hidden_states = hidden_states[:, -1, :].view(shape)
    lm_logits = self.lm_head(hidden_states).to(torch.float32)

    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        loss = loss.to(hidden_states.dtype)

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output
    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )
