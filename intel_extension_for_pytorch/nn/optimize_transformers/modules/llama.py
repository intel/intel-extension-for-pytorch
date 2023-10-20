import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union, List
from .transformer_modules.RoPE import LlamaRotaryEmbedding
from .transformer_modules.Norm import LlamaRMSNorm


from ._transformers import MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from .transformer_modules.Mlp import IPEXTransformerBaseMLP, IPEXTransformerMLPOptimizedFp16
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from .transformer_modules.QuantizedAttention import IPEXTransformerAttnNaive, IPEXTransformerAttnOptimizedFp16, IPEXTransformerAttnOptimizedInt4
from .transformer_modules.GroupedAttention import IPEXTransformerAttnOptimizedFp16Grouped
from transformers.modeling_outputs import CausalLMOutputWithPast
from .transformer_modules.Decoderblock import IPEXTransformerBlock
from .transformer_modules.Mlp import *
import sys

import os
acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in ["1", "ON", "Y", "YES", "TRUE"]

class NewIPEXLLAMABlock(IPEXTransformerBlock):
    def __init__(self,
                 module,
                 config,
                 dtype = "fp16",
                 device = "xpu",
                 module_name = "",
                 impl_mode = None,
                 tp_size = 1, 
                 tp_group = None):
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(config, device, dtype, impl_mode, tp_size, tp_group)
        self.attn = self.build_attention_from_config()
        self.mlp = self.build_mlp_from_config()
        self.input_layernorm = LlamaRMSNorm(self.ipex_config.embedding_dim, self.ipex_config.norm_eps)
        self.post_attn_layernorm = LlamaRMSNorm(self.ipex_config.embedding_dim, self.ipex_config.norm_eps)
        self.port_all_parameters_to_new_module()


    def build_attention_from_config(self):
        dtype = self.ipex_config.dtype
        impl = self.ipex_config.impl
        attn_type = IPEXTransformerAttn
        attn_type_str = "IPEXTransformerAttn"
        for elem in [impl.name, dtype, "Grouped"]:
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
        for elem in [impl.name, dtype, activation.name, "llama"]:
            mlp_type_str = mlp_type_str + elem.capitalize()
            if hasattr(sys.modules[__name__], mlp_type_str):
                mlp_type = getattr(sys.modules[__name__], mlp_type_str)
        return mlp_type(self.ipex_config)


    def build_ipex_transformer_config(self,
                                      config,
                                      device,
                                      dtype,
                                      impl_mode,
                                      tp_size,
                                      tp_group) -> IPEXTransformerConfig:
        activation_function = self.config.hidden_act
        ipex_activation = None
        for act in SupportedActivation:
            if activation_function in act.value:
                ipex_activation = act
                break
        assert ipex_activation is not None, "found unrecognized activation function, can not build ipex config from {}".format(activation_function)

        assert dtype in ["fp16", "int4"], "dtype tag {} passed to optimized_transformers is not supported!".format(dtype)

        return IPEXTransformerConfig(
            embedding_dim = self.config.hidden_size,
            intermediate_dim = self.config.intermediate_size,
            num_attention_head = self.config.num_attention_heads,
            # transformers==4.31.0
            num_key_value_head = self.config.num_key_value_heads,
            max_positions = max(self.config.max_position_embeddings, MAX_SEQ_LEN),
            max_out_positions = MAX_OUT_SEQ_LEN,
            rotary_embedding_class = LlamaRotaryEmbedding,
            rotary_dim = None,
            rotary_half=True,
            rotate_every_two=False,
            use_casual_mask = False,
            activation_function = self.config.hidden_act,
            ipex_act = ipex_activation,
            norm_eps = self.config.rms_norm_eps,
            residual_dropout = None,
            attn_dropout = None,
            enable_bias = False,
            residual_pdrop = None,
            scale_attention = True,
            is_decoder = False,
            do_norm_before = None,
            ln_elementwise_affine = None,
            positional_embedding_base = 10000,
            device = self.device,
            dtype = dtype,
            impl = impl_mode,
            tp_size = tp_size,
            tp_group = tp_group
        )

    def port_attn_parameter(self):
        # IPEXTransformerAttnOptimizedFp16
        self.attn.load_parameter(self.module.self_attn.q_proj, self.module.self_attn.k_proj, self.module.self_attn.v_proj, self.module.self_attn.o_proj)

    def port_mlp_parameter(self):
        # IPEXTransformerMLPOptimizedFp16SiluLlama
        self.mlp.load_parameter(self.module.mlp.gate_proj, self.module.mlp.down_proj, self.module.mlp.up_proj)

    def port_norm_parameter(self):
        self.input_layernorm.weight = self.module.input_layernorm.weight
        self.post_attn_layernorm.weight = self.module.post_attention_layernorm.weight

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
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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


