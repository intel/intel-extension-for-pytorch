import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, Union

from .Activation import ACT2FN
import os
import math
from dataclasses import dataclass

@dataclass
class IPEXRuntimeAttnCache:
    key_cache:   torch.Tensor = None
    value_cache: torch.Tensor = None
    key_prompt:  torch.Tensor = None
    value_prompt:torch.Tensor = None

    def clear_cache(self):
        self.key_cache    = None
        self.value_cache  = None
        self.key_prompt   = None
        self.value_prompt = None

class IPEXTransformerAttn(nn.Module):
    layer_id_static = 0
    batch_size = 1
    beam_size = 1
    timestamp = 0

    def __init__(self,
                 config) -> None:
        super().__init__()
        self.config = config
        self.layer_id = IPEXTransformerAttn.layer_id_static
        IPEXTransformerAttn.layer_id_static += 1

    @staticmethod
    def release_resources():
        raise NotImplementedError

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
        alibi: torch.Tensor = None,
        first_token = False,
        **kwargs
    ):
        self.pre_qkv(hidden_states=hidden_states, key_value_states=key_value_states, layer_past=layer_past, **kwargs)

        query, key, value = self.qkv_gemm(hidden_states, key_value_states, layer_past, **kwargs)

        query, key, value = self.post_qkv(query, key, value, position_ids, layer_past, **kwargs)

        present = self.get_present(query, key, value, use_cache)

        key, value = self.pre_sdp(key, value)

        attn_output, attn_weight = self.sdp(query, key, value, attention_mask, head_mask, alibi)

        attn_output = self.post_sdp(attn_output, residual)


        self.end_of_attention()
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weight, )
        else:
            outputs += (None, )
        return outputs

    def pre_qkv(self, hidden_states, layer_past):
        raise NotImplementedError

    @staticmethod
    def reset_timestamp():
        IPEXTransformerAttn.timestamp = 0

    def qkv_gemm(self, hidden_states, **kwargs):
        raise NotImplementedError


    def post_qkv(self, query, key, value, position_ids, **kwargs):
        raise NotImplementedError


    def get_present(self, query, key, value, use_cache):
        raise NotImplementedError


    def pre_sdp(self, key, value):
        raise NotImplementedError


    def sdp(self, query, key, value, attention_mask, head_mask, alibi):
        raise NotImplementedError


    def post_sdp(self, attn_output, residual=None):
        raise NotImplementedError


    def end_of_attention(self):
        if self.layer_id + 1 == self.layer_id_static:
            IPEXTransformerAttn.timestamp += 1

    def is_beam_search(self):
        return self.beam_size > 1

    def is_1st_token(self):
        return self.timestamp == 0

    def is_1st_token_beam_search(self):
        return self.is_beam_search() and self.is_1st_token()
