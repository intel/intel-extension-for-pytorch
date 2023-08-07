import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union

from intel_extension_for_pytorch.nn.utils._transformer_configuration import IPEXTransformerConfig
from .Activation import ACT2FN
from .RoPE import GPTJRotaryEmbedding
from ._transformers import IPEXTransformerAtten, IPEXTransformerMLP, IPEXTransformerConverter, MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from ._transformer_configuration import IPEXTransformerConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class IPEXGPTJAttn(IPEXTransformerAtten):
    def __init__(self, config) -> None:
        super().__init__(config)

class IPEXGPTJMLP(IPEXTransformerMLP):
    def __init__(self, config: IPEXTransformerConfig):
        super().__init__(config)

    def forward(self, hidden_states: Optional[torch.Tensor], attn_output, residual):
        if self.row_major:
            if isinstance(self.act, nn.GELU):
                hidden_states = torch.ops.torch_ipex.matmul_gelu(hidden_states, self.fc_in_wei, self.fc_in.bias, self.act.approximate)
            else:
                hidden_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.fc_in_wei, self.fc_in.bias)
                hidden_states = self.act(hidden_states)
            hidden_states = torch.ops.torch_ipex.mm_bias_resadd_resadd(hidden_states, self.fc_out_wei, self.fc_out.bias, attn_output, residual)
        else:
            hidden_states = self.fc_in(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.fc_out(hidden_states) + attn_output + residual
        return hidden_states

class IPEXGPTJBlock(nn.Module):
    def __init__(self, 
                 config:IPEXTransformerConfig):
        super().__init__()
        self.config = config
        self.config.intermediate_size = 4 * self.config.embed_dim if self.config.intermediate_size is None else self.config.intermediate_size
        self.attn = IPEXGPTJAttn(config)
        self.ln = nn.LayerNorm(self.config.embed_dim, eps=self.config.norm_eps)
        self.mlp = IPEXGPTJMLP(config)
    
    def release_resources(self):
        self.attn.release_resources()
        self.mlp.release_resources()

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False
    ) ->  Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        # hidden_states:  [bs*beam, seq, hidden_size]
        # position_ids:   [bs*beam, seq]
        # attention_mask: [bs*beam, head, q_seq, kv_seq]
        bs = IPEXTransformerAtten.batch_size
        dim = hidden_states.dim()
        if dim == 3:
            beam = hidden_states.shape[0] // bs
        elif dim == 4:
            beam = hidden_states.shape[1]
        else:
            print("Unsupported input shape")
            return
        seq = hidden_states.shape[-2] 
        hidden_size = hidden_states.shape[-1]
        hidden_shape = [bs, beam, seq, hidden_size]
        if seq > 1:
            hidden_states = hidden_states.view(hidden_shape)[:, 0, :, :]        # [bs, seq, hidden_size]
            if position_ids is not None:
                position_ids = position_ids.view(bs, beam, position_ids.shape[1])[:,0,:].view(bs, position_ids.shape[1])
            if attention_mask is not None:
                attention_mask = attention_mask.view(bs, beam, attention_mask.shape[1], attention_mask.shape[2], attention_mask.shape[3])[:,0,:,:,:].view(bs, attention_mask.shape[1], attention_mask.shape[2], attention_mask.shape[3])
        # convert layout form [bs, seq, hidden_size] to [seq, bs, hidden_size]
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        residual = hidden_states
        hidden_states = torch.ops.torch_ipex.fast_layer_norm(hidden_states, self.ln.normalized_shape, self.ln.weight, self.ln.bias, self.ln.eps)
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1: ]

        hidden_states = self.mlp(hidden_states, attn_output, residual)

        # convert hidden_states form [seq, beam, hidden_size] back to [beam, seq, hidden_size]
        hidden_states = hidden_states.transpose(0, 1)
        if seq > 1:
            hidden_states = hidden_states.view(bs, 1, seq, hidden_size).expand([bs, beam, seq, hidden_size])

        if use_cache:
            outputs = (hidden_states, ) + outputs
        else:
            outputs = (hidden_states, ) + outputs[1:]

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            hidden_states = hidden_states.reshape([-1, hidden_states.shape[-2], hidden_states.shape[-1]])
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
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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

class IPEXGPTJConverter(IPEXTransformerConverter):
    def __init__(self,
                 module,
                 config = None,
                 device = "cpu",
                 dtype = torch.float):
        from transformers.models.gptj.configuration_gptj import GPTJConfig
        super().__init__(module, config, device=device, dtype=dtype)
        self.config = config if config is not None else GPTJConfig()
        self.ipex_transformers_config = self.construct_transformer_config()
        # print(self.ipex_transformers_config.__dict__)
        self.ipex_optimized_module = self.construct_ipex_optimized_module()
        self.port_all_parameters_to_new_module()

    def construct_transformer_config(self):
        n_positions = max(self.config.n_positions, MAX_SEQ_LEN)
        embed_dim = self.config.n_embd
        num_head = self.config.n_head
        rotary_dim = self.config.rotary_dim
        activate_function = self.config.activation_function
        resid_pdrop = self.config.resid_pdrop
        attn_pdrop = self.config.attn_pdrop
        layer_norm_eps = self.config.layer_norm_epsilon
        use_cache = self.config.use_cache
        intermediate_size = self.config.n_inner
        return IPEXTransformerConfig(
            embed_dim=embed_dim,
            intermediate_size=intermediate_size,
            num_attention_heads=num_head,
            max_positions=n_positions,
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=GPTJRotaryEmbedding,
            rotary_dim=rotary_dim,
            rotate_half=False,
            rotate_every_two=True,
            use_casual_mask=True,
            activation_function=activate_function,
            norm_eps=layer_norm_eps,
            residual_dropout=resid_pdrop,
            attn_dropout=attn_pdrop,
            enable_bias=False,
            residual_pdrop=resid_pdrop,
            scale_attention=True,
            is_decoder=False,
            do_norm_before=False,
            ln_elementwise_affine=False,
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
        return IPEXGPTJBlock(self.ipex_transformers_config)

    def port_attn_parameters(self):
        if self.row_major:
            self.module.attn.q_proj.weight.data = self.module.attn.q_proj.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.attn.q_wei = self.module.attn.q_proj.weight
            self.ipex_optimized_module.attn.q_proj.bias = self.module.attn.q_proj.bias
            self.module.attn.k_proj.weight.data = self.module.attn.k_proj.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.attn.k_wei = self.module.attn.k_proj.weight
            self.ipex_optimized_module.attn.k_proj.bias = self.module.attn.k_proj.bias
            self.module.attn.v_proj.weight.data = self.module.attn.v_proj.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.attn.v_wei = self.module.attn.v_proj.weight
            self.ipex_optimized_module.attn.v_proj.bias = self.module.attn.v_proj.bias
            self.module.attn.out_proj.weight.data = self.module.attn.out_proj.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.attn.out_wei = self.module.attn.out_proj.weight
            self.ipex_optimized_module.attn.out_proj.bias = self.module.attn.out_proj.bias
            shape = [3, -1, self.module.attn.q_proj.weight.shape[-1]]
            self.ipex_optimized_module.attn.qkv_wei = torch.stack([self.ipex_optimized_module.attn.q_wei, self.ipex_optimized_module.attn.k_wei, self.ipex_optimized_module.attn.v_wei]).contiguous().view(shape)
            self.ipex_optimized_module.attn.q_wei.data = self.ipex_optimized_module.attn.qkv_wei[0, :, :]
            self.ipex_optimized_module.attn.k_wei.data = self.ipex_optimized_module.attn.qkv_wei[1, :, :]
            self.ipex_optimized_module.attn.v_wei.data = self.ipex_optimized_module.attn.qkv_wei[2, :, :]
            self.ipex_optimized_module.attn.qkv_bias = None
        else:
            self.ipex_optimized_module.attn.k_proj.weight = self.module.attn.k_proj.weight
            self.ipex_optimized_module.attn.k_proj.bias = self.module.attn.k_proj.bias
            self.ipex_optimized_module.attn.q_proj.weight = self.module.attn.q_proj.weight
            self.ipex_optimized_module.attn.q_proj.bias = self.module.attn.q_proj.bias
            self.ipex_optimized_module.attn.v_proj.weight = self.module.attn.v_proj.weight
            self.ipex_optimized_module.attn.v_proj.bias = self.module.attn.v_proj.bias
            self.ipex_optimized_module.attn.out_proj.weight = self.module.attn.out_proj.weight
            self.ipex_optimized_module.attn.out_proj.bias = self.module.attn.out_proj.bias

    def port_mlp_parameters(self):
        if self.row_major:
            self.module.mlp.fc_in.weight.data = self.module.mlp.fc_in.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_in_wei = self.module.mlp.fc_in.weight
            self.ipex_optimized_module.mlp.fc_in.bias = self.module.mlp.fc_in.bias
            self.module.mlp.fc_out.weight.data = self.module.mlp.fc_out.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_out_wei = self.module.mlp.fc_out.weight
            self.ipex_optimized_module.mlp.fc_out.bias = self.module.mlp.fc_out.bias
        else:
            self.ipex_optimized_module.mlp.fc_in.weight = self.module.mlp.fc_in.weight
            self.ipex_optimized_module.mlp.fc_in.bias = self.module.mlp.fc_in.bias
            self.ipex_optimized_module.mlp.fc_out.weight = self.module.mlp.fc_out.weight
            self.ipex_optimized_module.mlp.fc_out.bias = self.module.mlp.fc_out.bias

    def port_layer_norm_parameters(self):
        self.ipex_optimized_module.ln.weight = self.module.ln_1.weight
        self.ipex_optimized_module.ln.bias = self.module.ln_1.bias
    
    def port_all_parameters_to_new_module(self):
        self.port_attn_parameters()
        self.port_mlp_parameters()
        self.port_layer_norm_parameters()

    def get_transformed_module(self):
        return self.ipex_optimized_module      
