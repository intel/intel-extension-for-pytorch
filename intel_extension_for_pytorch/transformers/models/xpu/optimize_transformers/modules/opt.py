import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
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
from transformers.modeling_outputs import CausalLMOutputWithPast
from .transformer_modules.Decoderblock import IPEXTransformerBlock
from .transformer_modules.Mlp import *  # noqa
import sys
from torch.nn import CrossEntropyLoss
import os

acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in [
    "1",
    "ON",
    "Y",
    "YES",
    "TRUE",
]


class NewIPEXOPTBlock(IPEXTransformerBlock):
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

    def build_attention_from_config(self):
        dtype = self.ipex_config.dtype
        impl = self.ipex_config.impl
        attn_type = IPEXTransformerAttn
        attn_type_str = "IPEXTransformerAttn"
        for elem in [impl.name, dtype, "crossed"]:
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
        for elem in [impl.name, dtype, activation.name]:
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
            max_positions=max(self.config.max_position_embeddings, MAX_SEQ_LEN),
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=PositionalEmbedding,
            rotary_dim=None,
            rotary_half=False,
            rotate_every_two=False,
            use_casual_mask=False,
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
            tp_size=tp_size,
            tp_group=tp_group,
        )

    def port_attn_parameter(self):
        # IPEXTransformerAttnOptimizedFp16Opt IPEXTransformerAttnOptimizedFp16
        self.attn.load_parameter(
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
        self.attn.transpose_parameter()
        self.mlp.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()
        self.attn.cat_qkv()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
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
        first_token = True if acc_test or past_key_value is None else False
        hidden_size = hidden_states.shape[-1]
        hidden_shape = [bs, beam, seq, hidden_size]
        if first_token and beam > 1:
            # for 1st token, keep the original layout
            # reduce the duplicated info in beam dim
            # shape -> [bs*beam, seq, hidden_size]
            # layout -> [bs*beam, seq, hidden_size]
            hidden_states = hidden_states.view(hidden_shape)[:, 0, :, :].contiguous()
            if attention_mask is not None:
                shape = attention_mask.shape
                attention_mask = attention_mask.view(
                    bs, beam, shape[1], shape[2], shape[3]
                )[:, 0, :, :, :].view(bs, shape[1], shape[2], shape[3])

        else:
            # for 2nd to last token, we convert the layout
            # shape -> [bs*beam, seq, hidden_size]
            # convert layout form [bs*beam, seq, hidden_size] to [seq, bs*beam, hidden_size]
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, present_key_value, self_attn_weights = self.attn(
            hidden_states=hidden_states,
            layer_past=past_key_value,
            attention_mask=attention_mask,
            head_mask=layer_head_mask,
            output_attentions=output_attentions,
            residual=residual,
            first_token=first_token,
        )

        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        if first_token and beam > 1:
            # for 1st token, expand the result with beam
            hidden_states = hidden_states.view(bs, 1, seq, hidden_size)
            hidden_states = hidden_states.expand([bs, beam, seq, hidden_size])
        else:
            # for 2nd to last token, we convert the layout back
            # convert hidden_states form [seq, beam, hidden_size] back to [beam, seq, hidden_size]
            hidden_states = hidden_states.transpose(0, 1)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


def IPEXOPTForCausalLMForward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
            provide it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True`
        is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
            tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
            cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
            that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
            all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, OPTForCausalLM

    >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    >>> prompt = "Hey, are you consciours? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
    ```"""

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
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model.decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        head_mask=head_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    if hidden_states.dim() > 3:
        hidden_states = hidden_states.reshape(
            [-1, hidden_states.shape[-2], hidden_states.shape[-1]]
        )
    if not acc_test:
        shape = list(hidden_states.size())
        shape[1] = 1
        hidden_states = hidden_states[:, -1, :].view(shape)
    logits = self.lm_head(hidden_states).to(torch.float32)

    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(logits.device)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
