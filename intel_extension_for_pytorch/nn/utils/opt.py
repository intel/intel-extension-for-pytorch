import torch
import torch.nn as nn
from typing import Optional, Tuple

from intel_extension_for_pytorch.nn.utils._transformer_configuration import IPEXTransformerConfig
from ._transformers import IPEXTransformerAtten, IPEXTransformerMLP, IPEXTransformerConverter, MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from ._transformer_configuration import IPEXTransformerConfig
from .RoPE import PositionalEmbedding

class IPEXOptAtten(IPEXTransformerAtten):
    def __init__(self, 
                 config: IPEXTransformerConfig) -> None:
        super().__init__(config)
        self.scaling = self.head_dim**-0.5

    def compute_qkv(self,
                    hidden_states: torch.Tensor,
                    key_value_state: Optional[torch.Tensor] = None,
                    layer_past: Optional[Tuple[torch.Tensor]] = None):
        is_cross_attention = key_value_state is not None

        seq_len, bs, _ = hidden_states.size()
        if self.row_major:
            query_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.q_wei, self.q_proj.bias)
        else:
            query_states = self.q_proj(hidden_states) 
        query_states = query_states * self.scaling

        if is_cross_attention and layer_past is not None:
            key_states = layer_past[0]
            value_state = layer_past[1]
        elif is_cross_attention:
            if self.row_major:
                key_states = torch.ops.torch_ipex.matmul_bias_out(key_value_state, self.k_wei, self.k_proj.bias)
                value_state = torch.ops.torch_ipex.matmul_bias_out(key_value_state, self.v_wei, self.v_proj.bias)
            else:
                key_states = self.k_proj(key_value_state)
                value_state = self.v_proj(key_value_state)
        else:
            if self.row_major:
                key_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.k_wei, self.k_proj.bias)
                value_state = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.v_wei, self.v_proj.bias)
            else:
                key_states = self.k_proj(hidden_states)
                value_state = self.v_proj(hidden_states)

        query_states = query_states.view(seq_len, bs, self.num_attn_head, self.head_dim)
        key_states = key_states.view(seq_len, bs, self.num_attn_head, self.head_dim)
        value_state = value_state.view(seq_len, bs, self.num_attn_head, self.head_dim)
        return query_states, key_states, value_state


class IPEXOptMLP(IPEXTransformerMLP):
    def __init__(self, config: IPEXTransformerConfig):
        super().__init__(config)

    def forward(self, hidden_states, residual):
        if self.row_major:
            hidden_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.fc_in_wei, self.fc_in.bias)
            hidden_states = self.act(hidden_states)
            hidden_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.fc_out_wei, self.fc_out.bias)
            hidden_states += residual
        else:
            hidden_states = self.fc_out(self.act(self.fc_in(hidden_states)))
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
            residual=residual)

        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)

        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(0, 1)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value, )
        return outputs


class IPEXOptConverter(IPEXTransformerConverter):
    def __init__(self,
                 module,
                 config = None,
                 device = "cpu",
                 dtype = torch.float):
        from transformers.models.opt.configuration_opt import OPTConfig
        super().__init__(module, config, device=device, dtype=dtype)
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
            scale_attention=False,
            is_decoder=True,
            do_norm_before=do_layer_norm_before,
            ln_elementwise_affine=layer_norm_eltwise_affine,
            seq_first=True,
            kv_cache_optimize=False,
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
            self.ipex_optimized_module.attn.out_bias = self.module.self_attn.out_proj.bias
        else:
            self.ipex_optimized_module.attn.k_proj.weight = self.module.self_attn.k_proj.weight
            self.ipex_optimized_module.attn.k_proj.bias = self.module.self_attn.k_proj.bias
            self.ipex_optimized_module.attn.q_proj.weight = self.module.self_attn.q_proj.weight
            self.ipex_optimized_module.attn.q_proj.bias = self.module.self_attn.q_proj.bias
            self.ipex_optimized_module.attn.v_proj.weight = self.module.self_attn.v_proj.weight
            self.ipex_optimized_module.attn.v_proj.bias = self.module.self_attn.v_proj.bias
            self.ipex_optimized_module.attn.out_proj.weight = self.module.self_attn.out_proj.weight
            self.ipex_optimized_module.attn.out_proj.bias = self.module.self_attn.out_proj.bias

    def port_mlp_parameters(self):
        if self.row_major:
            self.module.fc1.weight.data = self.module.fc1.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_in_wei = self.module.fc1.weight.data
            self.ipex_optimized_module.mlp.fc_in.bias = self.module.fc1.bias
            self.module.fc2.weight.data = self.module.fc2.weight.transpose(0, 1).contiguous()
            self.ipex_optimized_module.mlp.fc_out_wei = self.module.fc2.weight
            self.ipex_optimized_module.mlp.fc_out.bias = self.module.fc2.bias
        else:
            self.ipex_optimized_module.mlp.fc_in.weight = self.module.fc1.weight
            self.ipex_optimized_module.mlp.fc_in.bias = self.module.fc1.bias
            self.ipex_optimized_module.mlp.fc_out.weight = self.module.fc2.weight
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
