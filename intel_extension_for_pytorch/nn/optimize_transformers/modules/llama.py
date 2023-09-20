import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union, List

from ._transformer_configuration import IPEXTransformerConfig
from ._transformers import IPEXTransformerAtten, IPEXTransformerMLP, IPEXEmptyLinear, IPEXTransformerConverter, MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from ._transformer_configuration import IPEXTransformerConfig
from .RoPE import LlamaRotaryEmbedding
from .Norm import LlamaRMSNorm
from transformers.modeling_outputs import CausalLMOutputWithPast
        
import os
acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in ["1", "ON", "Y", "YES", "TRUE"]


class IPEXLlamaAttn(IPEXTransformerAtten):
    def __init__(self, config) -> None:
        super().__init__(config)


class IPEXLlamaMLP(IPEXTransformerMLP):
    def __init__(self,
                 config: IPEXTransformerConfig):
        super().__init__(config)
        self.fc_gate = IPEXEmptyLinear()
        self.fc_gate_wei = None
        self.fc_gate_bias = None

    def forward(self, hidden_states, residual):
        if self.row_major:
            if isinstance(self.act, nn.SiLU):
                hidden_states1 = torch.ops.torch_ipex.mm_silu(hidden_states, self.fc_gate_wei)
            else:
                hidden_states1 = torch.matmul(hidden_states, self.fc_gate_wei)
                hidden_states1 = self.act(hidden_states1)
            hidden_states = torch.ops.torch_ipex.mm_resmul(hidden_states, self.fc_in_wei, hidden_states1)
            shape = list(hidden_states.size())
            shape[-1] = self.fc_out_wei.shape[-1]
            hidden_states = torch.addmm(residual.flatten(0, -2), hidden_states.flatten(0, -2), self.fc_out_wei, beta=1.0/self.tp_size).view(shape)
            hidden_states = self.all_reduce_if_necessary(hidden_states)
        else:
            hidden_states_up1 = self.act(self.fc_gate(hidden_states))
            hidden_states_up2 = self.fc_in(hidden_states)
            hidden_states = self.fc_out(hidden_states_up1 * hidden_states_up2)
            hidden_states = self.all_reduce_if_necessary(hidden_states)
            hidden_states += residual
        return hidden_states


class IPEXLlamaBlock(nn.Module):
    def __init__(self, 
                 config: IPEXTransformerConfig):
        super().__init__()
        self.attn = IPEXLlamaAttn(config=config)
        self.mlp = IPEXLlamaMLP(config=config)
        self.input_layernorm = LlamaRMSNorm(config.embed_dim, eps=config.norm_eps)
        self.post_attn_layernorm = LlamaRMSNorm(config.embed_dim, eps=config.norm_eps)
        col_major = os.environ.get("COL_MAJOR", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        self.row_major = not col_major

    def release_resources(self):
        self.attn.release_resources()
        self.mlp.release_resources()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
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
        first_token = True if acc_test or past_key_value is None else False
        hidden_size = hidden_states.shape[-1]
        hidden_shape = [bs, beam, seq, hidden_size]
        if self.row_major:
            if first_token and beam > 1:
                # for 1st token, keep the original layout
                # reduce the duplicated info in beam dim
                # shape -> [bs*beam, seq, hidden_size]
                # layout -> [bs*beam, seq, hidden_size]
                hidden_states = hidden_states.view(hidden_shape)[:, 0, :, :].contiguous()
                if position_ids is not None:
                    position_ids = position_ids.view(bs, beam, position_ids.shape[1])[:,0,:].view(bs, position_ids.shape[1])
                if attention_mask is not None:
                    attention_mask = attention_mask.view(bs, beam, attention_mask.shape[1], attention_mask.shape[2], attention_mask.shape[3])[:,0,:,:,:].view(bs, attention_mask.shape[1], attention_mask.shape[2], attention_mask.shape[3])
            else:
                # for 2nd to last token, we convert the layout
                # shape -> [bs*beam, seq, hidden_size]
                # convert layout form [bs*beam, seq, hidden_size] to [seq, bs*beam, hidden_size]
                hidden_states = hidden_states.transpose(0, 1).contiguous()

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, self_attn_weights = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            residual=residual,
            first_token = first_token
        )
        residual = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)
        if self.row_major:
            if first_token and beam > 1:
                # for 1st token, expand the result with beam
                hidden_states = hidden_states.view(bs, 1, seq, hidden_size).expand([bs, beam, seq, hidden_size])
            else:
                # for 2nd to last token, we convert the layout back
                # convert hidden_states form [seq, beam, hidden_size] back to [beam, seq, hidden_size]
                hidden_states = hidden_states.transpose(0, 1)

        outputs = (hidden_states, )
        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs


def IPEXLlamaForCausalLMForward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
        if not acc_test:
            shape = list(hidden_states.size())
            shape[1] = 1
            hidden_states = hidden_states[:, -1, :].view(shape)
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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


class IPEXLlamaConverter(IPEXTransformerConverter):
    def __init__(self,
                 module,
                 config = None,
                 device = "cpu",
                 dtype = torch.float,
                 name = ''):
        super().__init__(module, config, device=device, dtype=dtype, name=name)
        from transformers.models.llama.configuration_llama import LlamaConfig
        self.config = config if config is not None else LlamaConfig()
        self.ipex_transformers_config = self.construct_transformer_config()
        self.ipex_optimized_module = self.construct_ipex_optimized_module()
        self.port_all_parameters_to_new_module()

    def construct_transformer_config(self):
        n_positions = max(self.config.max_position_embeddings, MAX_SEQ_LEN)
        embed_dim = self.config.hidden_size
        num_head = self.config.num_attention_heads
        num_key_value_head = self.config.num_key_value_heads
        activate_function = self.config.hidden_act
        norm_eps = self.config.rms_norm_eps
        use_cache = self.config.use_cache
        intermediate_size = self.config.intermediate_size
        rope_scaling = self.config.rope_scaling
        return IPEXTransformerConfig(
            embed_dim=embed_dim,
            intermediate_size=intermediate_size,
            num_attention_heads=num_head,
            num_key_value_heads=num_key_value_head,
            max_positions=n_positions,
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=LlamaRotaryEmbedding,
            rotary_dim=None,
            rotate_half=True,
            rotate_every_two=False,
            rope_scaling = rope_scaling,
            use_casual_mask=False,
            activation_function=activate_function,
            norm_eps=norm_eps,
            residual_dropout=None,
            attn_dropout=None,
            enable_bias=False,
            residual_pdrop=None,
            scale_attention=True,
            is_decoder=False,
            do_norm_before=None,
            ln_elementwise_affine=None,
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
        return IPEXLlamaBlock(self.ipex_transformers_config)

    def port_attn_parameters(self):
        if self.row_major:
            self.module.self_attn.q_proj.weight.data = self.module.self_attn.q_proj.weight.t().contiguous()
            self.module.self_attn.k_proj.weight.data = self.module.self_attn.k_proj.weight.t().contiguous()
            self.module.self_attn.v_proj.weight.data = self.module.self_attn.v_proj.weight.t().contiguous()

            shape = [3, -1, self.module.self_attn.q_proj.weight.shape[-1]]
            if self.ipex_transformers_config.num_attention_heads == self.ipex_transformers_config.num_key_value_heads:
                self.ipex_optimized_module.attn.qkv_wei = torch.cat([self.module.self_attn.q_proj.weight.data, 
                                                                     self.module.self_attn.k_proj.weight.data, 
                                                                     self.module.self_attn.v_proj.weight.data], dim=0).view(shape)
                self.module.self_attn.q_proj.weight.data = self.ipex_optimized_module.attn.qkv_wei[0, :, :]
                self.module.self_attn.k_proj.weight.data = self.ipex_optimized_module.attn.qkv_wei[1, :, :]
                self.module.self_attn.v_proj.weight.data = self.ipex_optimized_module.attn.qkv_wei[2, :, :]
            self.ipex_optimized_module.attn.q_wei = self.module.self_attn.q_proj.weight
            self.ipex_optimized_module.attn.k_wei = self.module.self_attn.k_proj.weight
            self.ipex_optimized_module.attn.v_wei = self.module.self_attn.v_proj.weight

            self.ipex_optimized_module.attn.q_proj.bias = self.module.self_attn.q_proj.bias
            self.ipex_optimized_module.attn.k_proj.bias = self.module.self_attn.k_proj.bias
            self.ipex_optimized_module.attn.v_proj.bias = self.module.self_attn.v_proj.bias

            self.module.self_attn.o_proj.weight.data = self.module.self_attn.o_proj.weight.t().contiguous()
            self.ipex_optimized_module.attn.out_wei = self.module.self_attn.o_proj.weight
            self.ipex_optimized_module.attn.out_bias = self.module.self_attn.o_proj.bias
        else:
            self.ipex_optimized_module.attn.k_proj.weight = self.module.self_attn.k_proj.weight
            self.ipex_optimized_module.attn.k_proj.bias = self.module.self_attn.k_proj.bias
            self.ipex_optimized_module.attn.q_proj.weight = self.module.self_attn.q_proj.weight
            self.ipex_optimized_module.attn.q_proj.bias = self.module.self_attn.q_proj.bias
            self.ipex_optimized_module.attn.v_proj.weight = self.module.self_attn.v_proj.weight
            self.ipex_optimized_module.attn.v_proj.bias = self.module.self_attn.v_proj.bias
            self.ipex_optimized_module.attn.out_proj.weight = self.module.self_attn.o_proj.weight
            self.ipex_optimized_module.attn.out_proj.bias = self.module.self_attn.o_proj.bias


    def port_mlp_parameters(self):
        if self.row_major:
            self.module.mlp.gate_proj.weight.data = self.module.mlp.gate_proj.weight.t().contiguous()
            self.ipex_optimized_module.mlp.fc_gate_wei = self.module.mlp.gate_proj.weight
            self.ipex_optimized_module.mlp.fc_gate_bias = self.module.mlp.gate_proj.bias

            self.module.mlp.up_proj.weight.data = self.module.mlp.up_proj.weight.t().contiguous()
            self.ipex_optimized_module.mlp.fc_in_wei = self.module.mlp.up_proj.weight
            self.ipex_optimized_module.mlp.fc_in_bias = self.module.mlp.up_proj.bias

            self.module.mlp.down_proj.weight.data = self.module.mlp.down_proj.weight.t().contiguous()
            self.ipex_optimized_module.mlp.fc_out_wei = self.module.mlp.down_proj.weight
            self.ipex_optimized_module.mlp.fc_out_bias = self.module.mlp.down_proj.bias
        else:
            self.ipex_optimized_module.mlp.fc_gate.weight = self.module.mlp.gate_proj.weight
            self.ipex_optimized_module.mlp.fc_gate.bias = self.module.mlp.gate_proj.bias
            self.ipex_optimized_module.mlp.fc_in.weight = self.module.mlp.up_proj.weight
            self.ipex_optimized_module.mlp.fc_in.bias = self.module.mlp.up_proj.bias
            self.ipex_optimized_module.mlp.fc_out.weight = self.module.mlp.down_proj.weight
            self.ipex_optimized_module.mlp.fc_out.bias = self.module.mlp.down_proj.bias


    def port_layer_norm_parameters(self):
        self.ipex_optimized_module.input_layernorm.weight = self.module.input_layernorm.weight
        self.ipex_optimized_module.post_attn_layernorm.weight = self.module.post_attention_layernorm.weight

    def port_all_parameters_to_new_module(self):
        self.port_attn_parameters()
        self.port_mlp_parameters()
        self.port_layer_norm_parameters()

    def get_transformed_module(self):
        return self.ipex_optimized_module
