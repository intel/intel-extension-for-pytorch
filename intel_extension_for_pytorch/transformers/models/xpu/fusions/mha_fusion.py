import torch
from torch import nn
from ...reference.fusions.mha_fusion import RotaryEmbedding


class _IPEXRopeXPU(nn.Module):
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

    @classmethod
    def rotary_embedding(
        cls, query, key, sin, cos, rotary_dim, rotary_half, position_ids=None
    ):
        rotary_query = query[..., :rotary_dim]
        rotary_key = key[..., :rotary_dim]
        last_dim = rotary_query.size(-1)
        assert (
            sin.size(-1) * 2 == last_dim or sin.size(-1) == last_dim
        ), "rotary embedding only support sin and cos's last dim equals query's last dim or half of last dim"
        if sin.size(-1) * 2 == last_dim:
            sin = sin.repeat(1, 1, 2)
            cos = cos.repeat(1, 1, 2)
        sin = sin.expand(rotary_query.shape)
        cos = cos.expand(rotary_key.shape)
        if rotary_half:
            torch.ops.torch_ipex.apply_rotary_embedding_half_qk(
                rotary_query, rotary_key, sin, cos, rotary_query, rotary_key
            )
        else:
            torch.ops.torch_ipex.apply_rotary_embedding_two_qk(
                rotary_query, rotary_key, sin, cos, rotary_query, rotary_key
            )

        return query, key

    def forward(
        self,
        query,
        key,
        sin,
        cos,
        rotary_dim,
        rotary_half,
        postion_ids=None,
        seqlens=None,
    ):
        raise NotImplementedError


class _IPEXRMSNormXPU(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.weight = module.weight
        self.eps = module.eps

    @classmethod
    def apply(cls, hidden_states, shape, weight, eps):
        return torch.ops.torch_ipex.rms_norm(hidden_states, shape, weight, eps)[0]

    def forward(self, hidden_states):
        return torch.ops.torch_ipex.rms_norm(
            hidden_states, [hidden_states.size(-1)], self.weight, self.eps
        )[0]


class _IPEXFastLayerNormXPU(nn.Module):
    def __init__(self, normalized_shape, eps, weight, bias=None):
        super().__init__()
        self.weight = weight
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.bias = bias

    @classmethod
    def apply(cls, hidden_states, normalized_shape, weight, bias, eps):
        return torch.ops.torch_ipex.fast_layer_norm(
            hidden_states, normalized_shape, weight, bias, eps
        )

    def forward(self, hidden_states):
        return torch.ops.torch_ipex.fast_layer_norm(
            hidden_states, self.normalized_shape, self.weight, self.bias, self.eps
        )


class _IPEXVarlenScaledDotProductXPU(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def apply(
        cls,
        query,
        key,
        value,
        out,
        seqlen_q,
        seqlen_k,
        max_seqlen_q,
        max_seqlen_k,
        pdropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        return_softmax,
        gen_,
    ):
        return torch.xpu.varlen_fwd(
            query,
            key,
            value,
            out,
            seqlen_q,
            seqlen_k,
            max_seqlen_q,
            max_seqlen_k,
            pdropout,
            softmax_scale,
            zero_tensors,
            is_causal,
            return_softmax,
            gen_,
        )

    def forward(
        self,
        query,
        key,
        value,
        out,
        seqlen_q,
        seqlen_k,
        max_seqlen_q,
        max_seqlen_k,
        pdropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        return_softmax,
        gen_,
    ):
        return torch.xpu.varlen_fwd(
            query,
            key,
            value,
            out,
            seqlen_q,
            seqlen_k,
            max_seqlen_q,
            max_seqlen_k,
            pdropout,
            softmax_scale,
            zero_tensors,
            is_causal,
            return_softmax,
            gen_,
        )


class _IPEXPagedAttentionXPU:
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch.ops.torch_ipex.reshape_and_cache(
            key, value, key_cache, value_cache, slot_mapping
        )

    @classmethod
    def single_query_cached_kv_attention(
        cls,
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
    ):
        torch.ops.torch_ipex.xetla_paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            block_tables,
            context_lens,
            scale,
            block_size,
            max_context_len,
            alibi_slopes,
        )
