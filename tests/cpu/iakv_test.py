import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from common_utils import TestCase
import unittest
from typing import Optional, Tuple, Union
from torch.nn import functional as F
import time 

class MaskedMHA(torch.nn.Module):
    def __init__(self, hidden_size=4096, n_head=16, n_head_kv=16, head_dim = 256):
        super().__init__()
        self.num_heads = n_head
        self.num_kv = n_head_kv
        self.head_dim = head_dim
        self.query_key_value = nn.Linear(hidden_size, (n_head_kv * 2 + n_head) * head_dim)
        
    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory
        storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, (num_heads + kv_num * 2) * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim]
            key: [batch_size, seq_length, kv_num, head_dim]
            value: [batch_size, seq_length, kv_num, head_dim]
        """
        bs = fused_qkv.shape[0]
        query_layer = fused_qkv[:, :, : self.num_heads * self.head_dim]
        query_layer = query_layer.view(bs, -1, self.num_heads, self.head_dim)
        key_layer = fused_qkv[:, :, self.num_heads * self.head_dim : (self.num_heads + self.num_kv) * self.head_dim]
        key_layer = key_layer.view(bs, -1, self.num_kv, self.head_dim)
        value_layer = fused_qkv[:, :, (self.num_heads + self.num_kv) * self.head_dim :]
        value_layer = value_layer.view(bs, -1, self.num_kv, self.head_dim)
        return query_layer, key_layer, value_layer
    
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        "torch.repeat_interleave(x, dim=2, repeats=n_rep)"
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return(
            x[:,:,:,None,:]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )    
        
    def forward(self, input_t, key_cache, value_cache, max_position, attention_mask, beam_idx, indirect_access_kv_cache=True, offset=0):
        head_size= self.head_dim
        #self.query_key_value(input_t)
        #linear_res=  torch.randn(input_t.shape[0], input_t.shape[1], (self.num_heads + self.num_kv * 2) * head_size, dtype=torch.bfloat16)
        query = torch.randn(input_t.shape[0], input_t.shape[1], self.num_heads, self.head_dim, dtype=torch.bfloat16)
        key = query.clone()
        value = key.clone()
        return torch.ops.torch_ipex.masked_multihead_self_attention(query, key, value, key_cache, value_cache, beam_idx, offset, head_size**0.5, max_position, None, attention_mask)

mha = MaskedMHA()
max_seq_len=2048
head_num=16
beam_size=4
head_size=256
batch_size=1
input_t = torch.randn(batch_size*beam_size, 1, head_num * head_size, dtype=torch.bfloat16)
key_cache_iakv = torch.randn(max_seq_len, beam_size*batch_size, head_num, head_size, dtype=torch.bfloat16) 
value_cache_iakv = torch.randn(max_seq_len, beam_size*batch_size, head_num, head_size, dtype=torch.bfloat16)
beam_idx = torch.zeros(max_seq_len, beam_size*batch_size, dtype=torch.int64)   
offset = 2016 
attention_mask = torch.zeros(batch_size*beam_size, 1, 1, offset+1, dtype=torch.bfloat16)  
count =10000
total_time = 0
for i in range(count):
    start =time.time()
    with torch.inference_mode(), torch.no_grad(), torch.autocast(
                            device_type="cpu",
                            enabled=True,
                            dtype=torch.bfloat16,
                        ):
        indirect_access_kv_cache_output, _, key_cache_iakv, value_cache_iakv, beam_idx = mha(input_t, key_cache_iakv, value_cache_iakv, max_seq_len, attention_mask, beam_idx, True, torch.tensor(offset))
    end = time.time()
    if i>=5:
        total_time += end-start
print("iakv time: ", total_time/(count-5))
        
        
    