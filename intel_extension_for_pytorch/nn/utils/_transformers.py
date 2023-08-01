import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from intel_extension_for_pytorch.nn.utils._transformer_configuration import IPEXTransformerConfig
from .Activation import ACT2FN
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


class IPEXEmptyLinear(nn.Module):
    def __init__(self):
        super(IPEXEmptyLinear, self).__init__()
        # we set the weight and bias to None to avoid any possible memory presure
        self.weight = None
        self.bias = None

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, bias=self.bias)

class IPEXEmptyLinearWithPadding(nn.Module):
    def __init__(self, n_dim):
        super(IPEXEmptyLinearWithPadding, self).__init__()
        # we set the weight and bias to None to avoid any possible memory presure
        self.weight = None
        self.bias = None
        self.n_dim = n_dim

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, bias=self.bias)[:,:,:self.n_dim]

class IPEXTransformerAtten(nn.Module):

    layer_id_static = 0
    casual_attention_mask = None
    blocked_alibi = None
    blocked_attn_mask = None
    beam_index = None
    batch_size = 1
    runtime_bs = 0

    def __init__(self, config) -> None:
        super(IPEXTransformerAtten, self).__init__()
        self.config:IPEXTransformerConfig = config
        self.seq_first = self.config.seq_first
        self.kv_cache_optimize = self.config.kv_cache_optimize
        self.layer_id = IPEXTransformerAtten.layer_id_static
        self.max_positions = self.config.max_positions
        self.max_out_positions = self.config.max_out_positions
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
        self.kv_cache_invalid = True

        seq_first = os.environ.get("SEQ_FIRST", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        disable_kv_cache = os.environ.get("DISABLE_KV_CACHE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
        self.kv_cache = not disable_kv_cache

        self.is_decoder = self.config.is_decoder
        self.residual_drop = nn.Dropout(self.config.residual_dropout) if self.config.residual_dropout is not None else nn.Identity()
        self.attn_drop = nn.Dropout(self.config.attn_dropout) if self.config.attn_dropout is not None else nn.Identity() 
        if self.use_casual_mask:
            mask = torch.ones((self.max_positions, self.max_positions), dtype=torch.float)
            mask = (1 - torch.tril(mask).view(1, 1, self.max_positions, self.max_positions)) * (-66504.0)
            IPEXTransformerAtten.attention_mask = mask.to(self.config.device) 

        # the cached key/value for the input prompt
        self.key_prompt = None
        self.value_prompt = None
        self.prev_len = 0 
        self.cur_len = 0

    @staticmethod
    def update_beam_index(beam_index):
        IPEXTransformerAtten.beam_index = beam_index
    
    @staticmethod
    def release_all_static_cached_resources():
        IPEXTransformerAtten.casual_attention_mask = None
        IPEXTransformerAtten.blocked_alibi = None
        IPEXTransformerAtten.blocked_attn_mask = None
        IPEXTransformerAtten.beam_index = None
    
    def release_resources(self):
        self.key_cached = None
        self.value_cached = None
        self.key_prompt = None
        self.value_prompt = None

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
        # hidden_states has already been converted to [seq, beam, hidden_size]
        if hidden_states.shape[0] != 1:
            # the first timestep
            shape = [self.max_positions, hidden_states.shape[1], self.num_attn_head, self.head_dim]
            self.key_cached = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.value_cached = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.prev_len = 0
        elif self.key_cached is None or self.value_cached is None:
            prev_key, value_key = layer_past[0], layer_past[1]
            self.prev_len = prev_key.size(2)
            shape = [self.max_positions, hidden_states.shape[1], self.num_attn_head, self.head_dim]
            self.key_cached = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.value_cached = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            self.prev_len = layer_past[0].size(2)

        if not self.checking_cache(layer_past):
            self.prev_len = layer_past[0].size(2)
            self.key_cached[:self.prev_len, :, :, :] = layer_past[0].permute(2, 0, 1, 3)
            self.value_cached[:self.prev_len,: , :, :] = layer_past[1].permute(2, 0, 1, 3)

        self.cur_len = self.prev_len + hidden_states.size(0)

        shape = [hidden_states.shape[0], hidden_states.shape[1], self.num_attn_head * self.head_dim]
        query = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
        key = self.key_cached[self.prev_len : self.cur_len, :, :, :]
        value = self.value_cached[self.prev_len : self.cur_len, :, :, :]
        key = key.view(shape)
        value = value.view(shape)

        torch.ops.torch_ipex.mm_qkv_out(hidden_states, self.qkv_wei, self.qkv_bias, query, key, value)
        return query, key, value

    def qkv_cache_optimized_beam(self, hidden_states, layer_past = None):
        # beam search path
        # hidden_states has already been converted to [seq, bs*beam, hidden_size]
        if hidden_states.shape[0] != 1:
            # the first timestep
            # first timestamp's shape will be [seq, bs, hidden_size]
            shape = [hidden_states.shape[0], hidden_states.shape[1], self.num_attn_head * self.head_dim]
            query = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.key_prompt = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.value_prompt = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            torch.ops.torch_ipex.mm_qkv_out(hidden_states, self.qkv_wei, self.qkv_bias, query, self.key_prompt, self.value_prompt)
            key = self.key_prompt
            value = self.value_prompt
            self.prev_len = 0
            self.cur_len = 0
            if hidden_states.shape[1] != IPEXTransformerAtten.runtime_bs:
                self.kv_cache_invalid = True
        else:
            # the 2nd to the last timestep
            if self.key_cached is None or self.value_cached is None or self.kv_cache_invalid:
                # the 2nd generated token, create the key_cached and value_cached buffers
                shape = [self.max_out_positions, hidden_states.shape[1], self.num_attn_head, self.head_dim]
                self.key_cached = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
                self.value_cached = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
                self.prev_len = 0
                self.kv_cache_invalid = False
                if self.layer_id == IPEXTransformerAtten.layer_id_static - 1:
                    IPEXTransformerAtten.runtime_bs = hidden_states.shape[1]

            self.cur_len = self.prev_len + hidden_states.size(0)
            shape = [hidden_states.shape[0], hidden_states.shape[1], self.num_attn_head * self.head_dim]
            query = torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
            key = self.key_cached[self.prev_len : self.cur_len, :, :, :]
            value = self.value_cached[self.prev_len : self.cur_len, :, :, :]
            key = key.view(shape)
            value = value.view(shape)
            torch.ops.torch_ipex.mm_qkv_out(hidden_states, self.qkv_wei, self.qkv_bias, query, key, value)
        self.prev_len = self.cur_len
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
        if self.kv_cache_optimize and self.row_major and self.kv_cache:
            if self.get_beam_width() == 1:
                query, key, value = self.qkv_cache_optimized_greedy(hidden_states=hidden_state, layer_past=layer_past)
            else:
                # TODO: support greedy
                # update prompts status
                new_prompts = True if layer_past == None else False
                if new_prompts:
                    self.key_prompt = None
                    self.value_prompt = None
                query, key, value = self.qkv_cache_optimized_beam(hidden_states=hidden_state, layer_past=layer_past)
            new_shape = query.size()[:-1] + (self.num_attn_head, self.head_dim)
            if self.cur_len == 0:
                if self.key_prompt is not None:
                    self.key_prompt = self.key_prompt.view(new_shape)
                if self.value_prompt is not None:
                    self.value_prompt = self.value_prompt.view(new_shape)
        else:
            # qkv fusion now is only support on greedy search 
            query, key, value = self.qkv_normal(hidden_states=hidden_state, layer_past=layer_past)

        new_shape = query.size()[:-1] + (self.num_attn_head, self.head_dim)
        # reshape the qkv size from (seq_len, bs*beam, num_head*head_dim) to (seq_len, bs*beam, num_head, head_dim)
        query = query.view(new_shape)
        key = key.view(new_shape)
        value = value.view(new_shape)

        return query, key, value

    def optimized_combine(self, query, key, value, layer_past = None):
        if self.cur_len == 0:
            key = self.key_prompt
            value = self.value_prompt
            self.key_prompt = self.key_prompt.permute(1, 2, 0, 3)
            self.value_prompt = self.value_prompt.permute(1, 2, 0, 3)
        else:
            # query/key/value shape and layout [seq, bs*beam, num_head, head_dim]
            key = self.key_cached[: self.cur_len, :, :, :]  # [seq_len, bs*beam, head, head_dim]
            value = self.value_cached[: self.cur_len, :, :, :] # [seq_len, bs*beam, head, head_dim]
        query = query.permute(1, 2, 0, 3)  # [bs*beam, head, seq_len, head_dim]
        key = key.permute(1, 2, 0, 3)  # [bs*beam, head, seq_len, head_dim]
        value = value.permute(1, 2, 0, 3)  # [bs*beam, head, seq_len, head_dim]
        return query, key, value

    def normal_combine(self, query, key, value, layer_past = None):
        # query/key/value has been converted to [seq, bs*beam, num_head, head_dim]
        query = query.permute(1, 2, 0, 3)
        key = key.permute(1, 2, 0, 3)
        value = value.permute(1, 2, 0, 3)
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        return query, key, value

    def combine_with_cache(self, query, key, value, layer_past = None):
        if self.kv_cache_optimize and self.row_major and self.kv_cache:
            query, key, value = self.optimized_combine(query, key, value, layer_past=layer_past)
        else:
            query, key, value = self.normal_combine(query, key, value, layer_past=layer_past)
        return query, key, value

    def apply_rotary_embedding(self, key, query, position_ids):
        return self.position_emb(key, query, position_ids, self.layer_id)

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
            IPEXTransformerAtten.blocked_attn_mask.fill_(-65504.);
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

            if self.cur_len > 0 and self.get_beam_width() != 1:
                ########### 2nd token to the end
                # query shape [bs*beam, head, q_seq, dim], layout [q_seq, bs*beam, head, dim]
                # query shape [4, 16, 1, 256], stride [4096, 256, 16384, 1]
                # key_prompt shape [bs, head, prompt_seq, dim], layout [prompt_seq, bs, head, dim]
                # key_prompt shape [1, 16, 32, 256], stride [4096, 256, 4096, 1]
                # value_prompt is the same as key_prompt
                # key shape [bs*beam, head, kv_seq, dim], layout [kv_seq, bs*beam, head, dim]
                # key shape [4, 16, kv_len, 256]
                # value is the same as key
                # attention_mask= None
                # is_causal=False
                # beam_idx shape [kv_len, beam], layout [kv_len, beam], dtype=int32
                # self.cur_len = kv_len
                attn_output = torch.xpu.IpexSDP_Index(query, self.key_prompt, self.value_prompt, key, value, IPEXTransformerAtten.beam_index, blocked_alibi, blocked_attn_mask, head_mask, self.cur_len, alpha, beta, dropout, is_causal)
            else:
                ########### first token
                # query shape [bs*beam, head, q_seq, dim], layout [q_seq, bs*beam, head, dim]
                # query shape [1, 16, 32, 256], stride [4096, 256, 4096, 1]
                # key, value are the same as query
                # attention_mask= None
                # is_causal=True
                attn_output = torch.xpu.IpexSDP(query, key, value, blocked_alibi, blocked_attn_mask, head_mask, alpha, beta, dropout, is_causal, True)
        else:
            if self.cur_len > 0:
                key, value = self.reorder_cache(key, value, IPEXTransformerAtten.beam_index)
            attn_output, attn_weights = self.naive_self_attention(query, key, value, attention_mask=attention_mask, head_mask=head_mask, alibi=alibi)
        return attn_output, attn_weights

    def get_beam_width(self):
        return IPEXTransformerAtten.beam_index.shape[1] // IPEXTransformerAtten.batch_size if IPEXTransformerAtten.beam_index != None else 1

    def expand_beam_idx(self):
        bs = IPEXTransformerAtten.batch_size
        beam_idx = IPEXTransformerAtten.beam_index
        beam = beam_idx.shape[1] // bs
        expand_beam_idx = torch.empty_like(beam_idx)
        for i in range(bs):
            expand_beam_idx[:, i*beam:(i+1)*beam] = beam_idx[:, i*beam:(i+1)*beam]  + beam * i
        return expand_beam_idx

    def reorder_cache(self, key, value, beam_idx_cache):
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        # past_key_values: 28 decoder layers of [key, value]
        # beam_idx_cache: [kv_out_len, bs*beam]

        # self.key_prompt shape[bs, head, seq, dim] layout[seq, bs, head, dim]
        # key shape[bs*beam, head, kv_len, dim] layout[kv_len, bs*beam, head, dim]
        bs = self.key_prompt.shape[0]
        beam = int(key.shape[0] // bs)
        expand_shape = [bs, beam, self.key_prompt.shape[1]*self.key_prompt.shape[2]*self.key_prompt.shape[3]]
        shape = [bs*beam, self.key_prompt.shape[1], self.key_prompt.shape[2], self.key_prompt.shape[3]]
        #shape1 = [bs* beam, key.shape[1], key.shape[2], key.shape[3]]
        key_prompt = self.key_prompt.reshape(bs, 1, -1).expand(expand_shape).reshape(shape)
        value_prompt = self.value_prompt.reshape(bs, 1, -1).expand(expand_shape).reshape(shape)
        key_list = [key_prompt]
        value_list = [value_prompt]
        beam_idx_cache = self.expand_beam_idx()
        for idx in range(beam_idx_cache.shape[0]):
            beam_idx = beam_idx_cache[idx]
            current_key = key[:, :, idx, :].view(bs*beam, key.shape[1], 1, -1)
            current_key = current_key.index_select(0, beam_idx.to(key.device))
            key_list.append(current_key)
            current_value = value[:, :, idx, :].view(bs*beam, value.shape[1], 1, -1)
            current_value = current_value.index_select(0, beam_idx.to(value.device))
            value_list.append(current_value)

        key = torch.cat(key_list, dim=2)
        value = torch.cat(value_list, dim=2)
        return key, value

    def get_final_output(self, attn_output: torch.Tensor, residual: Optional[torch.Tensor] = None):
        # reshape the attn_output from [bs, head_num, seq_len, head_dim] back to [seq_len, bs, embedding_size]
        attn_output = attn_output.permute(2, 0, 1, 3)
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
            if residual is not None:
                attn_output += residual
        return attn_output

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
        # the shape of query, key, value, [seq, bs*beam, head*dim]
        # the layout of query, key, value, [seq, bs*beam, head*dim]
        query, key, value = self.compute_qkv(hidden_state=hidden_states, key_value_state=key_value_states, layer_past=layer_past)

        # the shape of query, key, value, [seq, bs*beam, head, dim]
        # the layout of query, key, value, [seq, bs*beam, head, dim]
        key, query = self.apply_rotary_embedding(key, query, position_ids=position_ids)

        # the shape of query, key, value, [seq, bs*beam, head, dim]
        # the layout of query, key, value, [seq, bs*beam, head, dim]
        query, key, value = self.combine_with_cache(query, key, value, layer_past=layer_past)

        if use_cache or self.is_decoder:
            present = (key, value)
        else:
            present = None

        # the shape of query, key, value, [bs*beam, head, seq, dim]
        # the layout of query, key, value, [seq, bs*beam, head, dim]
        attn_output, attn_weight = self.self_attention(query, key, value, attention_mask, head_mask, alibi)
        attn_output = self.get_final_output(attn_output=attn_output, residual=residual)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weight, )
        else:
            outputs += (None, )

        return outputs          # return as (attn_output, present, output_atten)


class IPEXTransformerMLP(nn.Module):
    batch_size = 1
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

    @staticmethod
    def release_resources():
        pass

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
                hidden_states = torch.ops.torch_ipex.matmul_gelu(hidden_states, self.fc_in_wei, self.fc_in.bias, self.act.approximate)
            else:
                hidden_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.fc_in_wei, self.fc_in.bias)
                hidden_states = self.act(hidden_states)
            hidden_states = torch.ops.torch_ipex.matmul_bias_out(hidden_states, self.fc_out_wei, self.fc_out.bias)
        else:
            hidden_states = self.fc_in(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.fc_out(hidden_states)
        return self.drop_out(hidden_states)
