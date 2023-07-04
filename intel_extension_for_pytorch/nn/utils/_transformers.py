import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from .RoPE import GPTJRotaryEmbedding, LlamaRotaryEmbedding, PositionalEmbedding
from ._transformer_configuration import IPEXTransformerConfig
import os

def activation_replace(module):
    from transformers.activations import NewGELUActivation

    replace_dict = {
        NewGELUActivation: torch.nn.GELU(approximate="tanh")
    }
    for m in replace_dict.keys():
        if isinstance(module, m):
            return replace_dict[m]
    return module


from collections import OrderedDict


ACT2CLS = {
    "gelu": nn.GELU(),
    "gelu_new": nn.GELU(approximate='tanh'),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
    "tanh": nn.Tanh()
}

ACT2FN = ACT2CLS

class IPEXTransformerAtten(nn.Module):

    layer_id_static = 0
    casual_attention_mask = None

    def __init__(self, config) -> None:
        super(IPEXTransformerAtten, self).__init__()
        self.config:IPEXTransformerConfig = config
        self.seq_first = self.config.seq_first
        self.kv_cache_optimize = self.config.kv_cache_optimize
        self.layer_id = IPEXTransformerAtten.layer_id_static
        self.max_positions = self.config.max_positions
        self.use_casual_mask = self.config.use_casual_mask
        self.layer_id = IPEXTransformerAtten.layer_id_static
        self.embed_dim = self.config.embed_dim
        self.num_attn_head = self.config.num_attention_heads
        IPEXTransformerAtten.layer_id_static += 1
        self.head_dim = self.config.embed_dim // self.config.num_attention_heads
        self.position_emb = self.config.rotary_embedding_class(config=self.config)
        if self.config.scale_attention:
            self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, device="xpu"))
        else:
            self.scale_attn = None
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.config.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.config.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.config.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.config.enable_bias)

        self.qkv_fused = True 
        self.q_wei = None
        self.k_wei = None
        self.v_wei = None
        self.out_wei = None
        self.qkv_wei = None 
        col_major = os.environ.get("COL_MAJOR", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        self.row_major = not col_major

        beam = os.environ.get("Beam", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        self.greedy = not beam

        self.is_decoder = self.config.is_decoder
        self.residual_drop = nn.Dropout(self.config.residual_dropout) if self.config.residual_dropout is not None else nn.Identity()
        self.attn_drop = nn.Dropout(self.config.attn_dropout) if self.config.attn_dropout is not None else nn.Identity() 
        if self.use_casual_mask:
            mask = torch.ones((self.max_positions, self.max_positions), dtype=torch.float)
            mask = (1 - torch.tril(mask).view(1, 1, self.max_positions, self.max_positions)) * (-66504.0)
            IPEXTransformerAtten.attention_mask = mask.to(self.config.device) 

    def qkv_cache_optimized_greedy(self, hidden_states, layer_past = None):
        # greedy search path
        if layer_past is None:
            # the first timestep
            shape = [hidden_states.shape[0], self.max_positions, self.num_attn_head, self.head_dim]
            self.key_cached = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.value_cached = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.prev_len = 0

        self.cur_len = self.prev_len + hidden_states.size(1)
        query = torch.empty_like(hidden_states)
        key = self.key_cached[:, self.prev_len : self.cur_len, :, :]
        value = self.value_cached[:, self.prev_len : self.cur_len, :, :]
        key = key.view(query.shape)
        value = value.view(query.shape)
        torch.ops.torch_ipex.mm_qkv_out(hidden_states, self.qkv_wei, query, key, value)
        return query, key, value

    def qkv_cache_optimized_beam(self, hidden_states, layer_past = None):
        # beam search path
        if layer_past is None:
            # the first timestep
            cached_shape = [self.max_positions, hidden_states.shape[0], self.num_attn_head * self.head_dim]
            self.key_cached = torch.empty(cached_shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.value_cached = torch.empty(cached_shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.prev_len = 0

        self.cur_len = self.prev_len + hidden_states.size(1)
        query = torch.empty_like(hidden_states)
        key_cached = self.key_cached[self.prev_len : self.cur_len, :, :].transpose(0, 1)
        value_cached = self.value_cached[self.prev_len : self.cur_len, :, :].transpose(0, 1)
        if key_cached.is_contiguous():
            key = key_cached
        else:
            key = torch.empty_like(hidden_states)
        if value_cached.is_contiguous():
            value = value_cached
        else:
            value = torch.empty_like(hidden_states)

        torch.ops.torch_ipex.mm_qkv_out(hidden_states, self.qkv_wei, query, key, value)

        self.key_cached[self.prev_len : self.cur_len, :, :] = key.transpose(0, 1)
        self.value_cached[self.prev_len : self.cur_len, :, :] = value.transpose(0, 1)
        return query, key, value

    def qkv_normal(self, hidden_states, layer_past = None):
        if self.row_major:
            query = torch.matmul(hidden_states, self.q_wei)
            key = torch.matmul(hidden_states, self.k_wei)
            value = torch.matmul(hidden_states, self.v_wei)
        else:
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
        return query, key, value


    def compute_qkv(self,
                    hidden_state: torch.Tensor,
                    key_value_state: Optional[torch.Tensor] = None,
                    layer_past: Optional[Tuple[torch.Tensor]] = None):
        if self.kv_cache_optimize and hidden_state.size(0) != 1:
            self.kv_cache_optimize = False
            print("Warning: kv cache optimize can only be enabled in greedy search. Set kv_cache_optimize to False !")
        if self.kv_cache_optimize:
            if self.greedy:
                query, key, value = self.qkv_cache_optimized_greedy(hidden_states=hidden_state, layer_past=layer_past)
            else:
                query, key, value = self.qkv_cache_optimized_beam(hidden_states=hidden_state, layer_past=layer_past)
        else:
            # qkv fusion now is only support on greedy search 
            query, key, value = self.qkv_normal(hidden_states=hidden_state, layer_past=layer_past)

        new_shape = query.size()[:-1] + (self.num_attn_head, self.head_dim)
        # reshape the qkv size to (bs,  seq_len, num_head, head_dim)
        query = query.view(new_shape)
        key = key.view(new_shape)
        value = value.view(new_shape)

        return query, key, value

    def optimized_combine(self, query, key, value, layer_past = None):
        if self.greedy:
            key = self.key_cached[:, : self.cur_len, :, :]  # [beam, seq_len, head, head_dim]
            value = self.value_cached[:, : self.cur_len, :, :]  # [beam, seq_len, head, head_dim]
            query = query.permute(0, 2, 1, 3) # [beam, head, seq_len, head_dim]
            key = key.permute(0, 2, 1, 3)  # [beam, head, seq_len, head_dim]
            value = value.permute(0, 2, 1, 3)  # [beam, head, seq_len, head_dim]
        else:
            shape = [self.cur_len, self.key_cached.shape[1], query.shape[2], query.shape[3]]
            key = self.key_cached[: self.cur_len, :, :].view(shape)  # [seq_len, beam, head*head_dim]
            value = self.value_cached[: self.cur_len, :, :].view(shape) # [seq_len, beam, head*head_dim]
            query = query.permute(0, 2, 1, 3)  # [beam, head, seq_len, head_dim]
            key = key.permute(1, 2, 0, 3)  # [beam, head, seq_len, head_dim]
            value = value.permute(1, 2, 0, 3)  # [beam, head, seq_len, head_dim]

        self.prev_len = self.cur_len

        return query, key, value

    def normal_combine(self, query, key, value, layer_past = None):
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        return query, key, value

    def combine_with_cache(self, query, key, value, layer_past = None):
        if self.kv_cache_optimize:
            query, key, value = self.optimized_combine(query, key, value, layer_past=layer_past)
        else:
            query, key, value = self.normal_combine(query, key, value, layer_past=layer_past)
        return query, key, value

    def apply_rotary_embedding(self, key, query, position_ids):
        return self.position_emb(key, query, position_ids)
        

    def naive_self_attention(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.use_casual_mask:
            # convert the casual mask dtype to target dtype, this should only happen once
            IPEXTransformerAtten.attention_mask.to(attn_weights.dtype)
            query_length, key_length = query.size(-2), key.size(-2)
            casual_mask = IPEXTransformerAtten.attention_mask[:, :, key_length - query_length : key_length, :key_length]
            # # TODO: Maybe we can move this line to the initializer
            # casual_mask *= -66504.0
            # replace torch.where as torch.add might helps with the host overhead
            attn_weights += casual_mask
        if self.scale_attn:
            attn_weights /= self.scale_attn

        if attention_mask is not None:
            attn_weights += attention_mask
            # the attn_weights should anyway bigger than dtype.min, I wonder if this is necessary
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float).to(query.dtype)
        attn_weights = self.attn_drop(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def self_attention(self, query, key, value, attention_mask=None, head_mask=None):
        do_sdp_fusion = os.environ.get("ENABLE_SDP_FUSION", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        if self.config.sdp_fusion_enable and do_sdp_fusion:
            if query.shape[2] <= 1:
                attn_output = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
            else:
                attn_output = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
            attn_weights = None
        else:
            attn_output, attn_weights = self.naive_self_attention(query, key, value, attention_mask=attention_mask, head_mask=head_mask)
        return attn_output, attn_weights

    def get_final_output(self, attn_output: torch.Tensor, residual: Optional[torch.Tensor] = None):
        # reshape the attn_output from [bs, head_num, seq_len, head_dim] back to [bs, seq_len, embedding_size]
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(attn_output.size()[:-2] + (self.embed_dim,))
        if self.row_major:
            if residual is None:
                attn_output = torch.matmul(attn_output, self.out_wei)
            else:
                shape = attn_output.shape
                attn_output = torch.addmm(residual.flatten(0, -2), attn_output.flatten(0, -2), self.out_wei)
                attn_output = attn_output.view(shape)
        else:
            attn_output = self.out_proj(attn_output) + residual
        return attn_output
        # return self.residual_drop(attn_output)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        key_value_states: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        residual: Optional[torch.Tensor] = None
    ):

        query, key, value = self.compute_qkv(hidden_state=hidden_states, key_value_state=key_value_states, layer_past=layer_past)

        key, query = self.apply_rotary_embedding(key, query, position_ids=position_ids)

        query, key, value = self.combine_with_cache(query, key, value, layer_past=layer_past)

        if use_cache or self.is_decoder:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weight = self.self_attention(query, key, value, attention_mask, head_mask)
        attn_output = self.get_final_output(attn_output=attn_output, residual=residual)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weight, )
        else:
            outputs += (None, )

        return outputs          # return as (attn_output, present, output_atten)

class IPEXGPTJAttn(IPEXTransformerAtten):
    def __init__(self, config) -> None:
        super().__init__(config)

class IPEXLlamaAttn(IPEXTransformerAtten):
    def __init__(self, config) -> None:
        super().__init__(config)

class IPEXOptAtten(IPEXTransformerAtten):
    def __init__(self, 
                 config: IPEXTransformerConfig) -> None:
        super().__init__(config)
        self.scaling = self.head_dim**-0.5

    def compute_qkv(self,
                    hidden_state: torch.Tensor,
                    key_value_state: Optional[torch.Tensor] = None,
                    layer_past: Optional[Tuple[torch.Tensor]] = None):
        is_cross_attention = key_value_state is not None

        bs, seq_len, _ = hidden_state.size()
        query_states = self.q_proj(hidden_state) 
        query_states = query_states * self.scaling

        if is_cross_attention and layer_past is not None:
            key_states = layer_past[0]
            value_state = layer_past[1]
        elif is_cross_attention:
            key_states = self.k_proj(key_value_state)
            value_state = self.v_proj(key_value_state)
        else:
            key_states = self.k_proj(hidden_state)
            value_state = self.v_proj(hidden_state)

        query_states = query_states.view(bs, seq_len, self.num_attn_head, self.head_dim)
        key_states = key_states.view(bs, seq_len, self.num_attn_head, self.head_dim)
        value_state = value_state.view(bs, seq_len, self.num_attn_head, self.head_dim)

        return query_states, key_states, value_state


class IPEXTransformerMLP(nn.Module):
    def __init__(self,
                 config: IPEXTransformerConfig):
        super().__init__()
        self.fc_in = nn.Linear(config.embed_dim, config.intermediate_size, bias=config.enable_bias)
        self.fc_out = nn.Linear(config.intermediate_size, config.embed_dim, bias=config.enable_bias)
        self.act = ACT2FN[config.activation_function]
        self.drop_out = nn.Dropout(config.residual_pdrop) if config.residual_pdrop is not None else nn.Identity()

        col_major = os.environ.get("COL_MAJOR", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        self.row_major = not col_major
        self.fc_in_wei = None
        self.fc_out_wei = None

    def forward(self, hidden_states: Optional[torch.Tensor]):
        if self.row_major:
            if isinstance(self.act, nn.GELU):
                hidden_states = torch.ops.torch_ipex.linear_gelu(hidden_states, self.fc_in.weight, self.fc_in.bias, self.act.approximate)
            else:
                hidden_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.fc_in_wei, self.fc_in.bias)
                hidden_states = self.act(hidden_states)
            hidden_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.fc_out_wei, self.fc_out.bias)
        else:
            hidden_states = self.fc_in(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.fc_out(hidden_states)
        return self.drop_out(hidden_states)

class IPEXGPTJMLP(IPEXTransformerMLP):
    def __init__(self, config: IPEXTransformerConfig):
        super().__init__(config)

    def forward(self, hidden_states: Optional[torch.Tensor], attn_output, residual):
        if self.row_major:
            if isinstance(self.act, nn.GELU):
                hidden_states = torch.ops.torch_ipex.linear_gelu(hidden_states, self.fc_in.weight, self.fc_in.bias, self.act.approximate)
            else:
                hidden_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.fc_in_wei, self.fc_in.bias)
                hidden_states = self.act(hidden_states)
            hidden_states = torch.ops.torch_ipex.mm_bias_resadd_resadd(hidden_states, self.fc_out_wei, self.fc_out.bias, attn_output, residual)
        else:
            hidden_states = self.fc_in(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.fc_out(hidden_states) + attn_output + residual
        return hidden_states

class IPEXLlamaMLP(IPEXTransformerMLP):
    def __init__(self,
                 config: IPEXTransformerConfig):
        super().__init__(config)
        self.up_proj = nn.Linear(config.embed_dim, config.intermediate_size, bias=config.enable_bias)
        self.up_wei = None

    def forward(self, hidden_states, residual):
        if self.row_major:
            if isinstance(self.act, nn.SiLU):
                hidden_states1 = torch.ops.torch_ipex.mm_silu(hidden_states, self.fc_in_wei)
            else:
                hidden_states1 = torch.matmul(hidden_states, self.fc_in_wei)
                hidden_states1 = self.act(hidden_states1)
            hidden_states = torch.ops.torch_ipex.mm_resmul(hidden_states, self.up_wei, hidden_states1)
            shape = list(hidden_states.size())
            shape[-1] = self.fc_out_wei.shape[-1]
            hidden_states = torch.addmm(residual.flatten(0, -2), hidden_states.flatten(0, -2), self.fc_out_wei).view(shape)
        else:
            hidden_states = self.fc_out(self.act(self.fc_in(hidden_states)) * self.up_proj(hidden_states))
            hidden_states += residual
        return hidden_states

class IPEXOptMLP(IPEXTransformerMLP):
    def __init__(self, config: IPEXTransformerConfig):
        super().__init__(config)


class IPEXGPTJBlock(nn.Module):
    def __init__(self, 
                 config:IPEXTransformerConfig):
        super().__init__()
        self.config = config
        self.config.intermediate_size = 4 * self.config.embed_dim if self.config.intermediate_size is None else self.config.intermediate_size
        self.attn = IPEXGPTJAttn(config)
        self.ln = nn.LayerNorm(self.config.embed_dim, eps=self.config.norm_eps)
        self.mlp = IPEXGPTJMLP(config)

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
        residual = hidden_states
        hidden_states, mean, var = torch.ops.torch_ipex.fast_layer_norm(hidden_states, self.ln.normalized_shape, self.ln.weight, self.ln.bias, self.ln.eps)
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
        if use_cache:
            outputs = (hidden_states, ) + outputs
        else:
            outputs = (hidden_states, ) + outputs[1:]

        return outputs

class LlamaRMSNorm(nn.Module):
    def __init__(self,
                 config: IPEXTransformerConfig):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.embed_dim))
        self.variance_epsilon = config.norm_eps

    def forward(self, hidden_states: torch.Tensor):
        hsz = hidden_states.shape[-1]
        hidden_states, mean, var = torch.ops.torch_ipex.fast_rms_norm(hidden_states, [hsz], self.weight, None, self.variance_epsilon)
        #output = torch.ops.torch_ipex.rms_norm(hidden_states, [hsz], self.weight)
        #return output[0]
        return hidden_states
        '''
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
        '''

class IPEXLlamaBlock(nn.Module):
    def __init__(self, 
                 config: IPEXTransformerConfig):
        super().__init__()
        self.attn = IPEXLlamaAttn(config=config)
        self.mlp = IPEXLlamaMLP(config=config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attn_layernorm = LlamaRMSNorm(config)

    def forward(
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

        hidden_states, present_key_value, self_attn_weights = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            residual=residual
        )

        residual = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, present_key_value, self_attn_weights = self.attn(
            hidden_states=hidden_states,
            layer_past=past_key_value,
            attention_mask=attention_mask,
            head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout_p, training=self.training)

        hidden_states = residual + hidden_states

        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value, )

        return outputs
