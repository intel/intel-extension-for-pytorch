import torch
from torch import nn
from typing import Optional, Tuple
from ...reference.fusions.mha_fusion import RotaryEmbedding


class _IPEXRopeCPU(nn.Module):
    def __init__(
        self,
        max_position_embeddings,
        pos_embd_dim,
        base=10000,
        backbone=None,
    ):
        super().__init__()
        self.embed_positions = RotaryEmbedding(
            max_position_embeddings, pos_embd_dim, backbone, base
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        num_head: int,
        head_dim: int,
        offset: int,
        rotary_ndims: int,
        seq_len: Optional[int] = None,
        num_concats: Optional[int] = None,
    ):
        position_ids = position_ids.contiguous()
        sin_cos, _, _ = self.embed_positions(seq_len)
        if num_concats is None:
            x, _, _ = torch.ops.torch_ipex.rotary_position_embedding(
                x,
                sin_cos,
                position_ids,
                num_head,
                head_dim,
                offset,
                rotary_ndims,
            )
            return x
        else:
            query, key, value = torch.ops.torch_ipex.rotary_position_embedding(
                x,
                sin_cos,
                position_ids,
                num_head,
                head_dim,
                offset,
                rotary_ndims,
            )
            return query, key, value


class _IPEXScaleDotProductCPU(nn.Module):
    def __init__(self, text_max_length):
        super().__init__()
        self.text_max_length = text_max_length

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_attn: float,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[Tuple[torch.Tensor]] = None,
        alibi: Optional[torch.Tensor] = None,
        add_casual_mask: Optional[bool] = True,
        seq_info: Optional[torch.Tensor] = None,
    ):
        if layer_past is None:
            layer_past = (
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros([1, 1, 1, 1]).contiguous(),
                torch.zeros([1, 1, 1, 1]).contiguous(),
                torch.zeros(1, int(query.size(0)), dtype=torch.long).contiguous(),
            )
        key_cache = layer_past[1].contiguous()
        value_cache = layer_past[2].contiguous()
        beam_idx = layer_past[3].contiguous()
        if seq_info is None:
            seq_info = torch.tensor(
                layer_past[0].size(-2), dtype=torch.long
            ).contiguous()
        (
            attn_output,
            attn_weights,
            key_cache,
            value_cache,
            beam_idx,
        ) = torch.ops.torch_ipex.masked_multihead_self_attention(
            query,
            key,
            value,
            key_cache,
            value_cache,
            beam_idx,
            seq_info,
            scale_attn,
            self.text_max_length,
            head_mask,
            attention_mask,
            add_casual_mask,
        )

        present = (
            torch.empty(
                1,
                (layer_past[0].size(-2) + query.shape[1]),
                (layer_past[0].size(-2) + query.shape[1]),
                1,
                dtype=torch.long,
            ).contiguous(),
            key_cache,
            value_cache,
            beam_idx,
        )
        return attn_output, attn_weights, present


class _IPEXRMSNorm(nn.Module):
    def __init__(self, module, config=None, tpp=False, woq=False):
        super().__init__()
        self.weight = module.weight
        if hasattr(module, "variance_epsilon"):
            self.variance_epsilon = module.variance_epsilon
        elif hasattr(module, "epsilon"):
            self.variance_epsilon = module.epsilon
        elif hasattr(module, "eps"):
            self.variance_epsilon = module.eps

    def forward(self, hidden_states):
        return torch.ops.torch_ipex.rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )
