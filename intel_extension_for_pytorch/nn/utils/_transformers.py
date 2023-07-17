import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from intel_extension_for_pytorch.nn.utils._transformer_configuration import IPEXTransformerConfig
from .RoPE import GPTJRotaryEmbedding, LlamaRotaryEmbedding, PositionalEmbedding
from ._transformer_configuration import IPEXTransformerConfig
import os
import math

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



class BloomGELU(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

ACT2CLS = {
    "gelu": nn.GELU(),
    "gelu_new": nn.GELU(approximate='tanh'),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
    "tanh": nn.Tanh(),
    "bloom_gelu": nn.GELU(approximate='tanh') 
}

ACT2FN = ACT2CLS

class IPEXEmptyLinear(nn.Module):
    def __init__(self):
        super(IPEXEmptyLinear, self).__init__()
        # we set the weight and bias to None to avoid any possible memory presure
        self.weight = None
        self.bias = None

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, bias=self.bias)

class IPEXTransformerAtten(nn.Module):

    layer_id_static = 0
    casual_attention_mask = None
    blocked_alibi = None
    blocked_attn_mask = None

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
        self.tp_size = self.config.tp_size
        self.tp_group = self.config.tp_group
        self.num_attn_head = self.num_attn_head // self.tp_size
        IPEXTransformerAtten.layer_id_static += 1
        self.head_dim = self.config.embed_dim // self.config.num_attention_heads
        self.position_emb = self.config.rotary_embedding_class(config=self.config)
        if self.config.scale_attention:
            self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, device=self.config.device))
        else:
            self.scale_attn = None
        self.k_proj = IPEXEmptyLinear()
        self.v_proj = IPEXEmptyLinear()
        self.q_proj = IPEXEmptyLinear()
        self.out_proj = IPEXEmptyLinear()

        self.qkv_fused = True 
        self.q_wei = None
        self.k_wei = None
        self.v_wei = None
        self.out_wei = None
        self.qkv_wei = None 
        self.qkv_bias = None
        self.out_bias = None
        col_major = os.environ.get("COL_MAJOR", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        self.row_major = not col_major
        self.key_cached = None
        self.value_cached = None

        beam = os.environ.get("Beam", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        self.greedy = not beam

        self.is_decoder = self.config.is_decoder
        self.residual_drop = nn.Dropout(self.config.residual_dropout) if self.config.residual_dropout is not None else nn.Identity()
        self.attn_drop = nn.Dropout(self.config.attn_dropout) if self.config.attn_dropout is not None else nn.Identity() 
        if self.use_casual_mask:
            mask = torch.ones((self.max_positions, self.max_positions), dtype=torch.float)
            mask = (1 - torch.tril(mask).view(1, 1, self.max_positions, self.max_positions)) * (-66504.0)
            IPEXTransformerAtten.attention_mask = mask.to(self.config.device) 
    
    def checking_cache(self, layer_past):
        acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in ["1", "ON", "Y", "YES", "TRUE"]
        if not acc_test:
            return True
        if layer_past is None:
            return True
        prev_key, prev_value = layer_past[0], layer_past[1]
        prev_key_len = prev_key.size(2)
        if not torch.equal(self.key_cached[:prev_key_len, :, :, :], prev_key):
            return False
        if not torch.equal(self.value_cached[:prev_key_len, :, :, :], prev_value):
            return False
        return True

    def qkv_cache_optimized_greedy(self, hidden_states, layer_past = None):

        # greedy search path
        if layer_past is None:
            # the first timestep
            shape = [self.max_positions, hidden_states.shape[0], self.num_attn_head, self.head_dim]
            self.key_cached = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.value_cached = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.prev_len = 0
        elif self.key_cached is None or self.value_cached is None:
            prev_key, value_key = layer_past[0], layer_past[1]
            self.prev_len = prev_key.size(2)
            shape = [self.max_positions, hidden_states.shape[0], self.num_attn_head, self.head_dim]
            self.key_cached = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.value_cached = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            self.prev_len = layer_past[0].size(2)

        if not self.checking_cache(layer_past):
            self.prev_len = layer_past[0].size(2)
            self.key_cached[:self.prev_len, :, :, :] = layer_past[0].permute(2, 0, 1, 3)
            self.value_cached[:self.prev_len,: , :, :] = layer_past[1].permute(2, 0, 1, 3)

        self.cur_len = self.prev_len + hidden_states.size(1)

        shape = [hidden_states.shape[1], hidden_states.shape[0], self.num_attn_head * self.head_dim]
        query = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
        
        key = self.key_cached[self.prev_len : self.cur_len, :, :, :]
        value = self.value_cached[self.prev_len : self.cur_len, :, :, :]

        key = key.view(shape)
        value = value.view(shape)

        torch.ops.torch_ipex.mm_qkv_out(hidden_states, self.qkv_wei, self.qkv_bias, query, key, value)
        return query, key, value

    def qkv_cache_optimized_beam(self, hidden_states, layer_past = None):
        # beam search path
        if layer_past is None:
            # the first timestep
            shape = [self.max_positions, hidden_states.shape[0], self.num_attn_head, self.head_dim]
            self.key_cached = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.value_cached = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.prev_len = 0

        self.cur_len = self.prev_len + hidden_states.size(1)

        shape = [hidden_states.shape[1], hidden_states.shape[0], self.num_attn_head * self.head_dim]
        query = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)

        key = self.key_cached[self.prev_len : self.cur_len, :, :, :]
        value = self.value_cached[self.prev_len : self.cur_len, :, :, :]

        key = key.view(shape)
        value = value.view(shape)

        torch.ops.torch_ipex.mm_qkv_out(hidden_states, self.qkv_wei, self.qkv_bias, query, key, value)
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
        # if self.kv_cache_optimize and hidden_state.size(0) != 1:
        #     self.kv_cache_optimize = False
        #     print("Warning: kv cache optimize can only be enabled in greedy search. Set kv_cache_optimize to False !")
        if self.kv_cache_optimize and self.row_major:
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
            key = self.key_cached[: self.cur_len, :, :, :]  # [seq_len, bs, head, head_dim]
            value = self.value_cached[: self.cur_len, :, :, :]  # [seq_len, bs, head, head_dim]
            query = query.permute(1, 2, 0, 3) # [bs, head, seq_len, head_dim]
            key = key.permute(1, 2, 0, 3)  # [bs, head, seq_len, head_dim]
            value = value.permute(1, 2, 0, 3)  # [bs, head, seq_len, head_dim]
        else:
            key = self.key_cached[: self.cur_len, :, :, :]  # [seq_len, bs*beam, head, head_dim]
            value = self.value_cached[: self.cur_len, :, :, :] # [seq_len, bs*beam, head, head_dim]
            query = query.permute(1, 2, 0, 3)  # [bs*beam, head, seq_len, head_dim]
            key = key.permute(1, 2, 0, 3)  # [bs*beam, head, seq_len, head_dim]
            value = value.permute(1, 2, 0, 3)  # [bs*beam, head, seq_len, head_dim]

        # self.prev_len = self.cur_len

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
        if self.kv_cache_optimize and self.row_major:
            query, key, value = self.optimized_combine(query, key, value, layer_past=layer_past)
        else:
            query, key, value = self.normal_combine(query, key, value, layer_past=layer_past)
        return query, key, value

    def apply_rotary_embedding(self, key, query, position_ids):
        return self.position_emb(key, query, position_ids)

    def all_reduce_if_necessary(self, reduce_target):
        if self.tp_group is not None:
            try:
                from deepspeed import comm as dist
            except ImportError as e:
                print("Can not find deepspeed. If you are using multi-tile feature, please make sure the deepspeed is correctly installed in your environment")
                return 
            dist.all_reduce(reduce_target, group=self.tp_group)
            return reduce_target
        else:
            return reduce_target

    def get_blocked_alibi(self, alibi):
        if self.layer_id == 0:
            shape = [alibi.shape[0], alibi.shape[1], self.max_positions] # [beam*num_head, q_len, kv_len]
            IPEXTransformerAtten.blocked_alibi = torch.empty(shape, device=alibi.device, dtype=alibi.dtype)
            kv_len = alibi.shape[2]
            IPEXTransformerAtten.blocked_alibi[:, :, 0 : kv_len] = alibi
        return IPEXTransformerAtten.blocked_alibi

    def get_blocked_attn_mask(self, attn_mask):
        if self.layer_id == 0:
            IPEXTransformerAtten.blocked_attn_mask = torch.empty((attn_mask.shape[0], attn_mask.shape[1], attn_mask.shape[2], self.max_positions), device=attn_mask.device, dtype=attn_mask.dtype)
            IPEXTransformerAtten.blocked_attn_mask[:, :, :, 0 : attn_mask.shape[3]] = attn_mask
        return IPEXTransformerAtten.blocked_attn_mask

    def naive_self_attention(self, query, key, value, attention_mask=None, head_mask=None, alibi : torch.Tensor=None):
        if alibi is not None:
            batch_size, num_heads, q_length, dim = query.shape
            _, _, kv_length, _ = key.shape
            batch1 = query.view(-1, q_length, dim)
            batch2 = key.view(-1, kv_length, dim).transpose(1, 2)
            matmul_result = alibi.baddbmm(
                batch1=batch1,
                batch2=batch2,
                beta=self.beta,
                alpha=self.inv_norm_factor,
            )

            # change view to [batch_size, num_heads, q_length, kv_length]
            attention_scores = matmul_result.view(batch_size, num_heads, q_length, kv_length)
            #input_dtype = attention_scores.dtype 
            #if input_dtype == torch.float16:
            #    attention_scores = attention_scores.to(torch.float)
            attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
            attention_probs = nn.functional.softmax(attn_weights, dim=-1)

            # [batch_size, num_heads, q_length, kv_length]
            attention_probs = self.attn_drop(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # matmul: [batch_size * num_heads, q_length, head_dim]
            attn_output = torch.matmul(attention_probs, value)
        else:
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

    def self_attention(self, query, key, value, attention_mask=None, head_mask=None, alibi=None):
        do_sdp_fusion = os.environ.get("ENABLE_SDP_FUSION", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        if self.config.sdp_fusion_enable and do_sdp_fusion:
            attn_weights = None

            # parameters
            dropout = 0.0
            alpha = 1.0 / math.sqrt(self.head_dim)
            beta = 1.0 # TODO: ignored by native
            is_causal = False
            if self.use_casual_mask == True and query.shape[2] != 1:
                is_causal = True

            blocked_attn_mask = None
            if attention_mask != None:
                # transform the attention_mask to casual mask if the attention_mask is in bool
                if attention_mask.dtype == torch.bool:
                    blocked_attn_mask = None
                    if query.shape[2] != 1:
                        is_causal = True
                else:
                    blocked_attn_mask = self.get_blocked_attn_mask(attention_mask)


            blocked_alibi = None
            if alibi != None:
                blocked_alibi = self.get_blocked_alibi(alibi)


            attn_output = torch.xpu.IpexSDP(query, key, value, blocked_alibi, blocked_attn_mask, head_mask, alpha, beta, dropout, is_causal)
        else:
            attn_output, attn_weights = self.naive_self_attention(query, key, value, attention_mask=attention_mask, head_mask=head_mask, alibi=alibi)
        return attn_output, attn_weights

    def get_final_output(self, attn_output: torch.Tensor, residual: Optional[torch.Tensor] = None):
        # reshape the attn_output from [bs, head_num, seq_len, head_dim] back to [bs, seq_len, embedding_size]
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(attn_output.size()[:-2] + (self.embed_dim // self.tp_size,))

        if self.row_major:
            if residual is None:
                attn_output = torch.matmul(attn_output, self.out_wei)
                self.all_reduce_if_necessary(attn_output)
                if self.out_bias is not None:
                    attn_output += self.out_bias
            else:
                shape = [attn_output.shape[0], attn_output.shape[1], self.embed_dim]
                if self.out_bias is not None:
                    attn_output = torch.ops.torch_ipex.mm_bias_resadd(attn_output, self.out_wei, self.out_bias, residual, 1.0/self.tp_size)
                else:
                    attn_output = torch.addmm(residual.flatten(0, -2), attn_output.flatten(0, -2), self.out_wei, beta=1.0/self.tp_size)
                attn_output = attn_output.view(shape)
                self.all_reduce_if_necessary(attn_output)
        else:
            attn_output = torch.matmul(attn_output, self.out_proj.weight.t())
            self.all_reduce_if_necessary(attn_output)
            attn_output += residual
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
        residual: Optional[torch.Tensor] = None,
        alibi: torch.Tensor = None
    ):

        query, key, value = self.compute_qkv(hidden_state=hidden_states, key_value_state=key_value_states, layer_past=layer_past)

        key, query = self.apply_rotary_embedding(key, query, position_ids=position_ids)

        query, key, value = self.combine_with_cache(query, key, value, layer_past=layer_past)

        if use_cache or self.is_decoder:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weight = self.self_attention(query, key, value, attention_mask, head_mask, alibi)
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
        if self.row_major:
            query_states = torch.matmul(hidden_state, self.q_wei)
        else:
            query_states = self.q_proj(hidden_state) 
        query_states = query_states * self.scaling

        if is_cross_attention and layer_past is not None:
            key_states = layer_past[0]
            value_state = layer_past[1]
        elif is_cross_attention:
            if self.row_major:
                key_states = torch.matmul(key_value_state, self.k_wei)
                value_state = torch.matmul(key_value_state, self.v_wei)
            else:
                key_states = self.k_proj(key_value_state)
                value_state = self.v_proj(key_value_state)
        else:
            if self.row_major:
                key_states = torch.matmul(hidden_state, self.k_wei)
                value_state = torch.matmul(hidden_state, self.v_wei)
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
        self.fc_in = IPEXEmptyLinear()
        self.fc_out = IPEXEmptyLinear()
        self.act = ACT2FN[config.activation_function]
        self.drop_out = nn.Dropout(config.residual_pdrop) if config.residual_pdrop is not None else nn.Identity()

        self.tp_size = config.tp_size
        self.tp_group = config.tp_group
        col_major = os.environ.get("COL_MAJOR", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        self.row_major = not col_major
        self.fc_in_wei = None
        self.fc_out_wei = None
        self.fc_in_bias = None
        self.fc_out_bias = None
    
    def all_reduce_if_necessary(self, target):
        if self.tp_group is not None:
            try:
                from deepspeed import comm as dist
            except ImportError as e:
                print("Can not find deepspeed. If you are using multi-tile feature, please make sure the deepspeed is correctly installed in your environment")
                return 
            dist.all_reduce(target, group=self.tp_group)
        return target
            

    def forward(self, hidden_states: Optional[torch.Tensor]):
        if self.row_major:
            if isinstance(self.act, nn.GELU):
                hidden_states = torch.ops.torch_ipex.linear_gelu(hidden_states, self.fc_in_wei.t(), self.fc_in.bias, self.act.approximate)
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
                hidden_states = torch.ops.torch_ipex.linear_gelu(hidden_states, self.fc_in_wei.transpose(0, 1), self.fc_in.bias, self.act.approximate)
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
        self.up_proj = IPEXEmptyLinear()
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

class IPEXBloomMLP(IPEXTransformerMLP):
    def __init__(self, config: IPEXTransformerConfig):
        super().__init__(config)

    def forward(self, hidden_states, residual: torch.Tensor):
        if self.row_major:
            if isinstance(self.act, nn.GELU):
                hidden_states = torch.ops.torch_ipex.linear_gelu(hidden_states, self.fc_in_wei.t(), self.fc_in.bias, self.act.approximate)
            else:
                hidden_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.fc_in_wei, self.fc_in.bias)
                hidden_states = self.act(hidden_states)
            hidden_states = torch.ops.torch_ipex.mm_bias_resadd(hidden_states, self.fc_out_wei, self.fc_out_bias, residual, 1.0/self.tp_size)
            output = self.all_reduce_if_necessary(hidden_states)
        else:
            intermediate_output = self.act(self.fc_in(hidden_states))
            output = torch.matmul(intermediate_output, self.fc_out.weight.t())
            output = self.all_reduce_if_necessary(output)
            output += self.fc_out.bias + residual
        return output


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


class IPEXBloomBlock(nn.Module):
    def __init__(self, 
                 config: IPEXTransformerConfig):
        super().__init__()
        self.config = config
        self.input_layernorm = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)
        self.self_attention = IPEXBloomAttn(config)
        self.post_attention_layernorm = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)
        self.mlp = IPEXBloomMLP(config)

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
        #layernorm_output = self.input_layernorm(hidden_states)
        layernorm_output, _, _ = torch.ops.torch_ipex.fast_layer_norm(hidden_states, self.input_layernorm.normalized_shape, self.input_layernorm.weight, self.input_layernorm.bias, self.input_layernorm.eps)
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
            alibi=alibi
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]
        #layernorm_output = self.post_attention_layernorm(attention_output)
        layernorm_output, _, _ = torch.ops.torch_ipex.fast_layer_norm(attention_output, self.post_attention_layernorm.normalized_shape, self.post_attention_layernorm.weight, self.post_attention_layernorm.bias, self.post_attention_layernorm.eps)
        if self.config.do_norm_before:
            redisual = layernorm_output
        else:
            residual = attention_output

        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output, ) + outputs
        else:
            outputs = (output, ) + outputs[1:]
        return outputs
