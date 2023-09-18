import math
import warnings
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union

from ._transformer_configuration import IPEXTransformerConfig
from ._transformers import IPEXTransformerAtten, IPEXTransformerMLP, IPEXEmptyLinear, IPEXTransformerConverter, MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from ._transformer_configuration import IPEXTransformerConfig
from .RoPE import PositionalEmbedding
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from .utils import print_rank_x

import os
acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in ["1", "ON", "Y", "YES", "TRUE"]

class IPEXBloomAttn(IPEXTransformerAtten):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.query_key_value = IPEXEmptyLinear()
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

    def qkv_normal(self, hidden_states, layer_past = None):
        if self.row_major:
            shape = [hidden_states.shape[0], hidden_states.shape[1], self.num_attn_head * self.head_dim]
            query = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            key = torch.empty_like(query)
            value = torch.empty_like(query)
            torch.ops.torch_ipex.mm_qkv_out(hidden_states, self.qkv_wei, self.qkv_bias, query, key, value)
        else:
            fused_qkv = self.query_key_value(hidden_states)
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_attn_head, 3, self.head_dim)
            query = fused_qkv[..., 0, :].reshape(batch_size, seq_length, -1)
            key = fused_qkv[..., 1, :].reshape(batch_size, seq_length, -1)
            value = fused_qkv[..., 2, :].reshape(batch_size, seq_length, -1)
        return query, key, value


class IPEXBloomMLP(IPEXTransformerMLP):
    def __init__(self, config: IPEXTransformerConfig):
        super().__init__(config)

    def forward(self, hidden_states, residual: torch.Tensor):
        if self.row_major:
            if isinstance(self.act, nn.GELU):
                hidden_states = torch.ops.torch_ipex.matmul_gelu(hidden_states, self.fc_in_wei, self.fc_in.bias, 1.0, self.act.approximate)
            else:
                hidden_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.fc_in_wei, self.fc_in.bias)
                hidden_states = self.act(hidden_states)
            # print_rank_x(0, "before mm resadd: {}".format(hidden_states))
            hidden_states = torch.ops.torch_ipex.mm_bias_resadd(hidden_states, self.fc_out_wei, self.fc_out_bias, 1.0/self.tp_size, residual, 1.0/self.tp_size)
            # print_rank_x(0, "before all reduce: {}".format(hidden_states))
            output = self.all_reduce_if_necessary(hidden_states)
        else:
            intermediate_output = self.act(self.fc_in(hidden_states))
            output = torch.matmul(intermediate_output, self.fc_out.weight.t())
            output = self.all_reduce_if_necessary(output)
            output += self.fc_out.bias + residual
        return output


class IPEXBloomBlock(nn.Module):
    def __init__(self, 
                 config: IPEXTransformerConfig):
        super().__init__()
        self.config = config
        self.input_layernorm = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)
        self.self_attention = IPEXBloomAttn(config)
        self.post_attention_layernorm = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)
        self.mlp = IPEXBloomMLP(config)
        col_major = os.environ.get("COL_MAJOR", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        self.row_major = not col_major

    def release_resources(self):                                                                              
        self.self_attention.release_resources()
        self.mlp.release_resources()

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
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
        first_token = True if acc_test or layer_past is None else False 
        hidden_size = hidden_states.shape[-1]
        hidden_shape = [bs, beam, seq, hidden_size]
        if self.row_major:
            if first_token and beam > 1:
                # for 1st token, keep the original layout
                # reduce the duplicated info in beam dim
                # shape -> [bs*beam, seq, hidden_size]
                # layout -> [bs*beam, seq, hidden_size]
                hidden_states = hidden_states.view(hidden_shape)[:, 0, :, :].contiguous()
                if attention_mask is not None:
                    attention_mask = attention_mask.view(bs, beam, attention_mask.shape[1], attention_mask.shape[2], attention_mask.shape[3])[:,0,:,:,:].view(bs, attention_mask.shape[1], attention_mask.shape[2], attention_mask.shape[3])
            else:
                # for 2nd to last token, we convert the layout
                # shape -> [bs*beam, seq, hidden_size]
                # convert layout form [bs*beam, seq, hidden_size] to [seq, bs*beam, hidden_size]
                hidden_states = hidden_states.transpose(0, 1).contiguous()
        
        # layernorm_output = torch.ops.torch_ipex.fast_layer_norm(hidden_states, self.input_layernorm.normalized_shape, self.input_layernorm.weight, self.input_layernorm.bias, self.input_layernorm.eps)
        # NOTE: fast_layer_norm has accuracy issue in Ipex auto tp
        layernorm_output = torch.ops.torch_ipex.fast_layer_norm(hidden_states, self.input_layernorm.normalized_shape, self.input_layernorm.weight, self.input_layernorm.bias, self.input_layernorm.eps)
        if self.config.do_norm_before:
            residual = layernorm_output
        else:
            residual = hidden_states
        attn_outputs = self.self_attention(
            hidden_states = layernorm_output,
            layer_past = layer_past,
            attention_mask = attention_mask,
            head_mask = head_mask,
            use_cache = use_cache,
            output_attentions = output_attentions,
            residual=residual,
            alibi=alibi,
            first_token=first_token
        )

        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        #layernorm_output = self.post_attention_layernorm(attention_output)
        # import torch.distributed as dist
        # if dist.get_rank() == 1:
        #     torch.save(attention_output, "./attn output.pt")
        #     torch.save(self.post_attention_layernorm.weight, "./layer_weight.pt")
        #     torch.save(self.post_attention_layernorm.bias, "./layer_bias.pt")
        #     # torch.save(attention_output, "./attn output.pt")
        # import os
        # os.abort()
        # layernorm_output = torch.ops.torch_ipex.fast_layer_norm(attention_output, self.post_attention_layernorm.normalized_shape, self.post_attention_layernorm.weight, self.post_attention_layernorm.bias, self.post_attention_layernorm.eps)
        layernorm_output = torch.ops.torch_ipex.fast_layer_norm(attention_output, self.post_attention_layernorm.normalized_shape, self.post_attention_layernorm.weight, self.post_attention_layernorm.bias, self.post_attention_layernorm.eps)
        if self.config.do_norm_before:
            redisual = layernorm_output
        else:
            residual = attention_output
        hidden_states = self.mlp(layernorm_output, residual)

        if self.row_major:
            if first_token and beam > 1:
                # for 1st token, expand the result with beam
                hidden_states = hidden_states.view(bs, 1, seq, hidden_size)
                hidden_states = hidden_states.expand([bs, beam, seq, hidden_size])
            else:
                # for 2nd to last token, we convert the layout back
                # convert hidden_states form [seq, beam, hidden_size] back to [beam, seq, hidden_size]
                hidden_states = hidden_states.transpose(0, 1)

        if use_cache:
            outputs = (hidden_states, ) + outputs
        else:
            outputs = (hidden_states, ) + outputs[1:]
        return outputs


def _convert_to_bloom_cache_ipex(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, seq_length, head_dim = past_key_value[0][0].shape

        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, seq_length, head_dim),
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )

def IPEXBloomForCausalLMForward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if hidden_states.dim() > 3:
            hidden_states = hidden_states.reshape([-1, hidden_states.shape[-2], hidden_states.shape[-1]])
        if not acc_test:
            shape = list(hidden_states.size())
            shape[1] = 1
            hidden_states = hidden_states[:, -1, :].view(shape)
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

class IPEXBloomConverter(IPEXTransformerConverter):
    def __init__(self, module, config, device="cpu", dtype=torch.float, name="") -> None:
        from transformers.models.bloom.configuration_bloom import BloomConfig
        super().__init__(module, config, device, dtype, name)
        self.config: BloomConfig = config if config is not None else BloomConfig()
        self.ipex_transformers_config = self.construct_transformer_config()
        self.ipex_optimized_module = self.construct_ipex_optimized_module()
        self.port_all_parameters_to_new_module()
        # self.module_name = name

    def construct_transformer_config(self):
        # bloom don't have n_position attribute, we set it as 2048 just like other LLM models.
        n_positions = max(2048, MAX_SEQ_LEN)
        embed_dim = self.config.hidden_size
        num_head = self.config.n_head
        hidden_dropout = self.config.hidden_dropout
        attention_dropout = self.config.attention_dropout
        before_norm = self.config.apply_residual_connection_post_layernorm
        # activate_function = self.config.hidden_act
        norm_eps = self.config.layer_norm_epsilon
        use_cache = self.config.use_cache
        intermediate_size = 4 * embed_dim
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
            activation_function="bloom_gelu",
            norm_eps=norm_eps,
            residual_dropout=None,
            attn_dropout=attention_dropout,
            enable_bias=False,
            residual_pdrop=None,
            scale_attention=True,
            is_decoder=False,
            do_norm_before=before_norm,
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
        return IPEXBloomBlock(self.ipex_transformers_config)

    def port_attn_parameters(self):
        tp_size = self.ipex_transformers_config.tp_size
        if self.row_major:
            embed_dim = self.config.hidden_size
            num_head = self.config.n_head // tp_size
            shape = [num_head, 3, -1, embed_dim]

            self.module.self_attention.query_key_value.weight.data = \
                self.module.self_attention.query_key_value.weight.view(shape).contiguous().transpose(0, 1).contiguous().view([3, -1, embed_dim]).transpose(1, 2).contiguous()
            self.ipex_optimized_module.self_attention.qkv_wei = \
                nn.Parameter(self.module.self_attention.query_key_value.weight)

            self.module.self_attention.query_key_value.bias.data = \
                self.module.self_attention.query_key_value.bias.view([num_head, 3, -1]).transpose(0, 1).contiguous().view([3, -1])
            self.ipex_optimized_module.self_attention.qkv_bias = \
                nn.Parameter(self.module.self_attention.query_key_value.bias)

            self.module.self_attention.dense.weight.data = \
                self.module.self_attention.dense.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.self_attention.out_wei = nn.Parameter(self.module.self_attention.dense.weight)
            self.module.self_attention.dense.bias.data = self.module.self_attention.dense.bias.data
            self.ipex_optimized_module.self_attention.out_bias = nn.Parameter(self.module.self_attention.dense.bias)
        else:
            self.ipex_optimized_module.self_attention.query_key_value.weight = nn.Parameter(self.module.self_attention.query_key_value.weight)
            self.ipex_optimized_module.self_attention.query_key_value.bias = nn.Parameter(self.module.self_attention.query_key_value.bias)
            self.ipex_optimized_module.self_attention.out_proj.weight = nn.Parameter(self.module.self_attention.dense.weight)
            self.ipex_optimized_module.self_attention.out_proj.bias = nn.Parameter(self.module.self_attention.dense.bias)


    def port_mlp_parameters(self):
        if self.row_major:
            self.module.mlp.dense_h_to_4h.weight.data = self.module.mlp.dense_h_to_4h.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_in_wei = nn.Parameter(self.module.mlp.dense_h_to_4h.weight)
            self.ipex_optimized_module.mlp.fc_in.bias = nn.Parameter(self.module.mlp.dense_h_to_4h.bias)

            self.module.mlp.dense_4h_to_h.weight.data = self.module.mlp.dense_4h_to_h.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_out_wei = nn.Parameter(self.module.mlp.dense_4h_to_h.weight)
            self.module.mlp.dense_4h_to_h.bias.data = self.module.mlp.dense_4h_to_h.bias.data
            self.ipex_optimized_module.mlp.fc_out_bias = nn.Parameter(self.module.mlp.dense_4h_to_h.bias)
        else:
            self.ipex_optimized_module.mlp.fc_in.weight = nn.Parameter(self.module.mlp.dense_h_to_4h.weight)
            self.ipex_optimized_module.mlp.fc_in.bias = nn.Parameter(self.module.mlp.dense_h_to_4h.bias)
            self.ipex_optimized_module.mlp.fc_out.weight = nn.Parameter(self.module.mlp.dense_4h_to_h.weight)
            self.ipex_optimized_module.mlp.fc_out.bias = nn.Parameter(self.module.mlp.dense_4h_to_h.bias)

    def port_layer_norm_parameters(self):
        self.ipex_optimized_module.input_layernorm.weight = nn.Parameter(self.module.input_layernorm.weight)
        self.ipex_optimized_module.input_layernorm.bias = nn.Parameter(self.module.input_layernorm.bias)
        self.ipex_optimized_module.post_attention_layernorm.weight = nn.Parameter(self.module.post_attention_layernorm.weight)
        self.ipex_optimized_module.post_attention_layernorm.bias = nn.Parameter(self.module.post_attention_layernorm.bias)

    def port_all_parameters_to_new_module(self):
        self.port_attn_parameters()
        self.port_mlp_parameters()
        self.port_layer_norm_parameters()
        # self.print_all_param_with_name()

    # def print_all_param_with_name(self):
    #     import torch.distributed as dist
    #     if dist.get_rank() == 0:
    #         for name, param in self.ipex_optimized_module.named_parameters():
    #             name = self.module_name + "." + name
    #             print("name: ", name)
    #             print("para: ", param)

    def get_transformed_module(self):
        return self.ipex_optimized_module
