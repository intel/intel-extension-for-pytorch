import torch
import torch.nn as nn

from .._transformer_configuration import IPEXTransformerConfig
from .Attention import IPEXTransformerAttnOptimizedFp16


class IPEXTransformerAttnOptimizedFp16Grouped(IPEXTransformerAttnOptimizedFp16):
    def __init__(self, config: IPEXTransformerConfig) -> None:
        super().__init__(config)
        self.num_kv_head = config.num_key_value_head
        self.num_kv_group = self.num_attn_head // self.num_kv_head


    # grouped attention do not support fused_qkv computation
    def cat_qkv(self):
        if self.num_kv_head == self.num_attn_head:
            super().cat_qkv()

    def prepare_kv_cache(self, hidden_states):
        bs_beam, seq_len, _ = self.get_runtime_shape(hidden_states)
        batch_size= bs_beam // self.beam_size

        kv_shape = [self.max_position, self.batch_size * self.beam_size, self.num_kv_head, self.head_dim]
        if self.runtime_cache.key_cache is None or self.runtime_cache.value_cache is None or batch_size != self.batch_size:
            self.runtime_cache.key_cache = torch.empty(kv_shape, device=hidden_states.device, dtype=hidden_states.dtype)
            self.runtime_cache.value_cache = torch.empty(kv_shape, device=hidden_states.device, dtype=hidden_states.dtype)

    def prepare_qkv_input_1st_token_beam_search(self, hidden_states, **kwargs):
        bs_beam, seq_len, embed_dim = self.get_runtime_shape(hidden_states)
        kv_shape = [bs_beam, seq_len, self.num_kv_head * self.head_dim]
        query_shape = [bs_beam, seq_len, embed_dim]
        query = torch.empty(query_shape, device=hidden_states.device, dtype=hidden_states.dtype)
        return query, self.runtime_cache.key_prompt.view(kv_shape), self.runtime_cache.value_prompt.view(kv_shape)


    def prepare_qkv_input_2nd2last(self, hidden_states, **kwargs):
        bs_beam, seq_len, embed_dim = self.get_runtime_shape(hidden_states)
        query_shape = [seq_len, bs_beam, embed_dim]
        kv_shape = [seq_len, bs_beam, self.num_kv_head * self.head_dim]
        query = torch.empty(query_shape, device=hidden_states.device, dtype=hidden_states.dtype)
        key = self.runtime_cache.key_cache[self.prev_seq_len:self.seq_len, :, :, :].view(kv_shape)
        value = self.runtime_cache.value_cache[self.prev_seq_len:self.seq_len, :, :, :].view(kv_shape)
        return query, key, value

    def compute_qkv_gemm(self, hidden_states, query, key, value):
        if self.num_kv_group <= 1:
            return super().compute_qkv_gemm(hidden_states, query, key, value)
        hidden_states_flat = hidden_states.flatten(0, -2)
        if self.q_proj.bias is None:
            torch.matmul(hidden_states, self.q_proj.weight, out=query)
        else:
            torch.addmm(self.q_proj.bias, hidden_states_flat, self.q_proj.weight, out=query.flatten(0, -2))

        if self.k_proj.bias is None:
            torch.matmul(hidden_states, self.k_proj.weight, out=key)
        else:
            torch.addmm(self.k_proj.bias, hidden_states_flat, self.k_proj.weight, out=key.flatten(0, -2))

        if self.v_proj.bias is None:
            torch.matmul(hidden_states, self.v_proj.weight, out=value)
        else:
            torch.addmm(self.v_proj.bias, hidden_states_flat, self.v_proj.weight, out=value.flatten(0, -2))
        return query, key, value

    def process_qkv_output(self, hidden_states, query, key, value):
        query_shape = hidden_states.size()[:-1] + (self.num_attn_head, self.head_dim)
        kv_shape = hidden_states.size()[:-1] + (self.num_kv_head, self.head_dim)
        if self.is_1st_token_beam_search():
            self.runtime_cache.key_prompt = self.runtime_cache.key_prompt.view(kv_shape)
            self.runtime_cache.value_prompt = self.runtime_cache.value_prompt.view(kv_shape)
        if self.is_beam_search():
            self.prev_seq_len = self.seq_len
        query = query.view(query_shape)
        key = key.view(kv_shape)
        value = value.view(kv_shape)
        return query, key, value
