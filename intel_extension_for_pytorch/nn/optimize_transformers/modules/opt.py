import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from ._transformer_configuration import IPEXTransformerConfig
from ._transformers import IPEXTransformerAtten, IPEXTransformerMLP, IPEXTransformerConverter, MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from ._transformer_configuration import IPEXTransformerConfig
from .RoPE import PositionalEmbedding
from transformers.modeling_outputs import CausalLMOutputWithPast

class IPEXOptAtten(IPEXTransformerAtten):
    def __init__(self, 
                 config: IPEXTransformerConfig) -> None:
        super().__init__(config)

    def compute_qkv(self,
                    hidden_states: torch.Tensor,
                    key_value_state: Optional[torch.Tensor] = None,
                    layer_past: Optional[Tuple[torch.Tensor]] = None,
                    first_token=False):
        is_cross_attention = key_value_state is not None
        if is_cross_attention and layer_past is not None:
            if self.row_major:
                query = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.q_wei, self.q_proj.bias)
            else:
                query = self.q_proj(hidden_states)
            key = layer_past[0]
            value = layer_past[1]
        elif is_cross_attention:
            if self.row_major:
                query = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.q_wei, self.q_proj.bias)
                key = torch.ops.torch_ipex.matmul_bias_out(key_value_state, self.k_wei, self.k_proj.bias)
                value = torch.ops.torch_ipex.matmul_bias_out(key_value_state, self.v_wei, self.v_proj.bias)
            else:
                query = self.q_proj(hidden_states)
                key = self.k_proj(key_value_state)
                value = self.v_proj(key_value_state)
        else:
            new_shape = hidden_states.size()[:-1] + (self.num_attn_head, self.head_dim)
            if self.kv_cache_optimize and self.kv_cache:
                if IPEXTransformerAtten.beam_size == 1:
                    query, key, value = self.qkv_cache_optimized_greedy(hidden_states=hidden_states, layer_past=layer_past)
                else:
                    new_prompts = True if layer_past == None else False
                    if new_prompts:
                        self.key_prompt = None
                        self.value_prompt = None
                    query, key, value = self.qkv_cache_optimized_beam(hidden_states=hidden_states, first_token=first_token, layer_past=layer_past)
                    if first_token:
                        if self.key_prompt is not None:
                            self.key_prompt = self.key_prompt.view(new_shape)
                        if self.value_prompt is not None:
                            self.value_prompt = self.value_prompt.view(new_shape)
            else:
                query, key, value = self.qkv_normal(hidden_states=hidden_states, layer_past=layer_past)

        # reshape the qkv size
        # 1st token: from (bs*beam, seq_len, num_head*head_dim) to (bs*beam, seq_len, num_head, head_dim)
        # 2nd to last token: from (seq_len, bs*beam, num_head*head_dim) to (seq_len, bs*beam, num_head, head_dim)
        query = query.view(new_shape)
        key = key.view(new_shape)
        value = value.view(new_shape)
        return query, key, value


class IPEXOptMLP(IPEXTransformerMLP):
    def __init__(self, config: IPEXTransformerConfig):
        super().__init__(config)

    def forward(self, hidden_states, residual):
        if self.row_major:
            hidden_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.fc_in_wei, self.fc_in.bias)
            hidden_states = self.act(hidden_states)
            hidden_states = torch.ops.torch_ipex.mm_bias_resadd(hidden_states, self.fc_out_wei, self.fc_out.bias, 1.0, residual, 1.0/self.tp_size)
            hidden_states = self.all_reduce_if_necessary(hidden_states)
        else:
            hidden_states = torch.ops.torch_ipex.mm_bias_resadd(hidden_states, self.fc_out_wei, self.fc_out.bias, 1.0, residual, 1.0/self.tp_size)
            hidden_states = self.all_reduce_if_necessary(hidden_states)
            if residual is not None:
                hidden_states += residual
        return hidden_states

class IPEXOptBlock(nn.Module):
    def __init__(self,
                 config: IPEXTransformerConfig):
        super().__init__()
        self.attn = IPEXOptAtten(config=config)
        self.mlp = IPEXOptMLP(config=config)
        self.do_layer_norm_before = config.do_norm_before
        self.self_attn_layer_norm = nn.LayerNorm(config.embed_dim, elementwise_affine=config.ln_elementwise_affine)
        self.final_layer_norm = nn.LayerNorm(config.embed_dim, elementwise_affine=config.ln_elementwise_affine)
        self.dropout_p = config.residual_pdrop

    def release_resources(self):
        self.attn.release_resources()
        self.mlp.release_resources()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # hidden_states:  [bs*beam, seq, hidden_size]
        # position_ids:   [bs*beam, seq]
        # attention_mask: [bs*beam, head, q_seq, kv_seq]
        bs = IPEXTransformerAtten.batch_size
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
        IPEXTransformerAtten.beam_size = beam
        first_token = True if seq > 1 else False
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
                attention_mask = attention_mask.view(bs, beam, shape[1], shape[2], shape[3])[:,0,:,:,:].view(bs, shape[1], shape[2], shape[3])

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
            first_token=first_token)

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
            outputs += (present_key_value, )
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

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
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

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        hidden_states = hidden_states.reshape([-1, hidden_states.shape[-2], hidden_states.shape[-1]])
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
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

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

class IPEXOptConverter(IPEXTransformerConverter):
    def __init__(self,
                 module,
                 config = None,
                 device = "cpu",
                 dtype = torch.float,
                 name = ''):
        from transformers.models.opt.configuration_opt import OPTConfig
        super().__init__(module, config, device=device, dtype=dtype, name=name)
        self.config = config if config is not None else OPTConfig()
        self.ipex_transformers_config = self.construct_transformer_config()
        self.ipex_optimized_module = self.construct_ipex_optimized_module()
        self.port_all_parameters_to_new_module()

    def construct_transformer_config(self):
        n_positions = max(self.config.max_position_embeddings, MAX_SEQ_LEN)
        embed_dim = self.config.hidden_size
        num_head = self.config.num_attention_heads
        activate_function = self.config.activation_function
        resid_pdrop = self.config.dropout
        use_cache = self.config.use_cache
        intermediate_size = self.config.ffn_dim
        do_layer_norm_before = self.config.do_layer_norm_before
        enable_bias = self.config.enable_bias
        layer_norm_eltwise_affine = self.config.layer_norm_elementwise_affine
        # is_decoder = self.config.is_decoder
        return IPEXTransformerConfig(
            embed_dim=embed_dim,
            intermediate_size=intermediate_size,
            num_attention_heads=num_head,
            max_positions=n_positions,
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=PositionalEmbedding,
            rotary_dim=None,
            rotate_half=False,
            rotate_every_two=False,
            use_casual_mask=False,
            activation_function=activate_function,
            norm_eps=None,
            residual_dropout=resid_pdrop,
            attn_dropout=None,
            enable_bias=enable_bias,
            residual_pdrop=resid_pdrop,
            scale_attention=True,
            is_decoder=True,
            do_norm_before=do_layer_norm_before,
            ln_elementwise_affine=layer_norm_eltwise_affine,
            seq_first=True,
            kv_cache_optimize=True,
            positional_embedding_base=10000,
            sdp_fusion_enable=True,
            device=self.device,
            dtype=self.dtype,
            tp_size=IPEXTransformerConverter.tp_size,
            tp_group=IPEXTransformerConverter.tp_group
        )

    def construct_ipex_optimized_module(self):
        return IPEXOptBlock(self.ipex_transformers_config)

    def port_attn_parameters(self):
        if self.row_major:
            self.module.self_attn.q_proj.weight.data = self.module.self_attn.q_proj.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.attn.q_wei = self.module.self_attn.q_proj.weight
            self.ipex_optimized_module.attn.q_proj.bias = self.module.self_attn.q_proj.bias

            self.module.self_attn.k_proj.weight.data = self.module.self_attn.k_proj.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.attn.k_wei = self.module.self_attn.k_proj.weight
            self.ipex_optimized_module.attn.k_proj.bias = self.module.self_attn.k_proj.bias

            self.module.self_attn.v_proj.weight.data = self.module.self_attn.v_proj.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.attn.v_wei = self.module.self_attn.v_proj.weight
            self.ipex_optimized_module.attn.v_proj.bias = self.module.self_attn.v_proj.bias

            self.module.self_attn.out_proj.weight.data = self.module.self_attn.out_proj.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.attn.out_wei = self.module.self_attn.out_proj.weight
            if self.module.self_attn.out_proj.bias is not None:
                self.module.self_attn.out_proj.bias = nn.Parameter(self.module.self_attn.out_proj.bias / self.tp_size)
                self.ipex_optimized_module.attn.out_bias = self.module.self_attn.out_proj.bias

            shape = [3, -1, self.module.self_attn.q_proj.weight.shape[-1]]
            self.ipex_optimized_module.attn.qkv_wei = torch.stack([
                self.ipex_optimized_module.attn.q_wei, 
                self.ipex_optimized_module.attn.k_wei, 
                self.ipex_optimized_module.attn.v_wei]).contiguous().view(shape)
            self.ipex_optimized_module.attn.q_wei.data = self.ipex_optimized_module.attn.qkv_wei[0,:, :]
            self.ipex_optimized_module.attn.k_wei.data = self.ipex_optimized_module.attn.qkv_wei[1,:, :]
            self.ipex_optimized_module.attn.v_wei.data = self.ipex_optimized_module.attn.qkv_wei[2,:, :]

            self.ipex_optimized_module.attn.qkv_bias = torch.stack([
                self.ipex_optimized_module.attn.q_proj.bias, 
                self.ipex_optimized_module.attn.k_proj.bias, 
                self.ipex_optimized_module.attn.v_proj.bias]).contiguous().view([3, -1])
            self.ipex_optimized_module.attn.q_proj.bias.data = self.ipex_optimized_module.attn.qkv_bias[0, :]
            self.ipex_optimized_module.attn.k_proj.bias.data = self.ipex_optimized_module.attn.qkv_bias[1, :]
            self.ipex_optimized_module.attn.v_proj.bias.data = self.ipex_optimized_module.attn.qkv_bias[2, :]
        else:
            self.ipex_optimized_module.attn.k_proj.weight = self.module.self_attn.k_proj.weight
            self.ipex_optimized_module.attn.k_proj.bias = self.module.self_attn.k_proj.bias
            self.ipex_optimized_module.attn.q_proj.weight = self.module.self_attn.q_proj.weight
            self.ipex_optimized_module.attn.q_proj.bias = self.module.self_attn.q_proj.bias
            self.ipex_optimized_module.attn.v_proj.weight = self.module.self_attn.v_proj.weight
            self.ipex_optimized_module.attn.v_proj.bias = self.module.self_attn.v_proj.bias
            self.ipex_optimized_module.attn.out_proj.weight = self.module.self_attn.out_proj.weight
            if self.module.self_attn.out_proj.bias is not None:
                self.module.self_attn.out_proj.bias = nn.Parameter(self.module.self_attn.out_proj.bias / self.tp_size)
                self.ipex_optimized_module.attn.out_bias = self.module.self_attn.out_proj.bias

    def port_mlp_parameters(self):
        if self.row_major:
            self.module.fc1.weight.data = self.module.fc1.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_in_wei = self.module.fc1.weight.data
            self.ipex_optimized_module.mlp.fc_in.bias = self.module.fc1.bias
            self.module.fc2.weight.data = self.module.fc2.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_out_wei = self.module.fc2.weight
            if self.module.fc2.bias is not None:
                self.module.fc2.bias = nn.Parameter(self.module.fc2.bias / self.tp_size)
                self.ipex_optimized_module.mlp.fc_out.bias = self.module.fc2.bias
        else:
            self.ipex_optimized_module.mlp.fc_in.weight = self.module.fc1.weight
            self.ipex_optimized_module.mlp.fc_in.bias = self.module.fc1.bias
            self.ipex_optimized_module.mlp.fc_out.weight = self.module.fc2.weight
            if self.module.fc2.bias is not None:
                self.module.fc2.bias = nn.Parameter(self.module.fc2.bias / self.tp_size)
                self.ipex_optimized_module.mlp.fc_out.bias = self.module.fc2.bias

    def port_layer_norm_parameters(self):
        self.ipex_optimized_module.self_attn_layer_norm.weight = self.module.self_attn_layer_norm.weight
        self.ipex_optimized_module.self_attn_layer_norm.bias = self.module.self_attn_layer_norm.bias
        self.ipex_optimized_module.final_layer_norm.weight = self.module.final_layer_norm.weight
        self.ipex_optimized_module.final_layer_norm.bias = self.module.final_layer_norm.bias

    def port_all_parameters_to_new_module(self):
        self.port_attn_parameters()
        self.port_mlp_parameters()
        self.port_layer_norm_parameters()

    def get_transformed_module(self):
        return self.ipex_optimized_module
