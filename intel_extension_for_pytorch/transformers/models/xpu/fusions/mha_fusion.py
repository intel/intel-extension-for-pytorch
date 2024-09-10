import torch
from torch import nn
from ...reference.fusions.mha_fusion import RotaryEmbedding
from typing import Optional, Dict


class _IPEXRopeXPU(nn.Module):
    def __init__(
        self,
        max_position_embeddings,
        pos_embd_dim,
        base=10000,
        backbone=None,
        kwargs=None,
    ):
        super().__init__()
        self.embed_positions = RotaryEmbedding(
            max_position_embeddings,
            pos_embd_dim,
            backbone,
            base,
            device="xpu",
            kwargs=kwargs,
        )
        self.max_seqlen_cached = self.embed_positions.max_seq_len_cached
        self.emb_dim = self.embed_positions.emb.size(-1)
        self.cos_interleave_cached = torch.repeat_interleave(
            self.embed_positions.cos_cached[:, :, :, : self.emb_dim // 2], 2, 3
        )
        self.sin_interleave_cached = torch.repeat_interleave(
            self.embed_positions.sin_cached[:, :, :, : self.emb_dim // 2], 2, 3
        )

    def get_sin_cos(self, seqlen, offset):
        _, sin, cos = self.embed_positions()
        if offset == 1:
            if seqlen is not None and seqlen > self.max_seqlen_cached:
                self.max_seqlen_cached = seqlen
                self.cos_interleave_cached = torch.repeat_interleave(
                    self.embed_positions.cos_cached[:, :, :, : self.emb_dim // 2], 2, 3
                )
                self.sin_interleave_cached = torch.repeat_interleave(
                    self.embed_positions.sin_cached[:, :, :, : self.emb_dim // 2], 2, 3
                )
            return self.sin_interleave_cached, self.cos_interleave_cached
        return sin, cos

    def apply_embedding(self, query, sin, cos, offset, key=None):
        if key is not None and key.size() == query.size():
            sin = sin.expand(query.size())
            cos = cos.expand(query.size())
            rope_kernel = (
                torch.ops.torch_ipex.apply_rotary_embedding_two_qk
                if offset == 1
                else torch.ops.torch_ipex.apply_rotary_embedding_half_qk
            )
            rope_kernel(query, key, sin, cos, query, key)
            return query, key
        rope_kernel = (
            torch.ops.torch_ipex.apply_rotary_embedding_two
            if offset == 1
            else torch.ops.torch_ipex.apply_rotary_embedding_half
        )
        rope_kernel(query, sin.expand(query.size()), cos.expand(query.size()), query)
        if key is not None:
            rope_kernel(key, sin.expand(key.size()), cos.expand(key.size()), key)
        return query, key

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
        num_head = rotary_query.size(-2)
        num_kv_head = rotary_key.size(-2)
        if num_head == num_kv_head:
            sin = sin.expand(rotary_query.shape)
            cos = cos.expand(rotary_query.shape)
            if rotary_half:
                torch.ops.torch_ipex.apply_rotary_embedding_half_qk(
                    rotary_query, rotary_key, sin, cos, rotary_query, rotary_key
                )
            else:
                torch.ops.torch_ipex.apply_rotary_embedding_two_qk(
                    rotary_query, rotary_key, sin, cos, rotary_query, rotary_key
                )
        else:
            sin_q = sin.expand(rotary_query.shape)
            cos_q = cos.expand(rotary_query.shape)
            sin_k = sin.expand(rotary_key.shape)
            cos_k = cos.expand(rotary_key.shape)
            if rotary_half:
                torch.ops.torch_ipex.apply_rotary_embedding_half(
                    rotary_query, sin_q, cos_q, rotary_query
                )
                torch.ops.torch_ipex.apply_rotary_embedding_half(
                    rotary_key, sin_k, cos_k, rotary_key
                )
            else:
                torch.ops.torch_ipex.apply_rotary_embedding_two(
                    rotary_query, sin_q, cos_q, rotary_query
                )
                torch.ops.torch_ipex.apply_rotary_embedding_two(
                    rotary_key, sin_k, cos_k, rotary_key
                )

        return query, key

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
        def expand_sin_and_cos(x, sin, cos):
            return sin.expand(x.shape), cos.expand(x.shape)

        sin, cos = self.get_sin_cos(seq_len, offset)
        bs = x.size(0)
        seqlen = x.size(1)
        sin = sin.squeeze()[position_ids].unsqueeze(2).to(x.dtype)
        cos = cos.squeeze()[position_ids].unsqueeze(2).to(x.dtype)
        if num_concats is None:
            hidden_size = num_head * head_dim
            if hidden_size != x.size(2):
                kv_head = x.size(2) // head_dim
                x = x.view(bs, seqlen, kv_head, head_dim)
            else:
                x = x.view(bs, seqlen, num_head, head_dim)

            # sin,cos = expand_sin_and_cos(x, sin, cos)
            return self.apply_embedding(x, sin, cos, offset)[0]

        elif num_concats == 2:
            hidden_size = num_head * head_dim
            kv_head = num_head
            if hidden_size * 2 != x.size(2):
                kv_head = (x.size(2) - hidden_size) // head_dim
            q = x[:, :, :hidden_size].view(bs, seqlen, num_head, head_dim)
            k = x[:, :, hidden_size:].view(bs, seqlen, kv_head, head_dim)
            # sin, cos = expand_sin_and_cos(q, sin, cos)
            return self.apply_embedding(q, sin, cos, offset, k)

        elif num_concats == 3:
            hidden_size = num_head * head_dim
            kv_head = num_head
            if num_head * head_dim * 3 != x.size(2):
                kv_head = (x.size(2) - num_head * head_dim) // head_dim // 2
            kv_size = kv_head * head_dim
            q = x[:, :, :hidden_size].view(bs, seqlen, num_head, head_dim)
            k = x[:, :, hidden_size : hidden_size + kv_size].view(
                bs, seqlen, kv_head, head_dim
            )
            v = x[:, :, hidden_size + kv_size :].view(bs, seqlen, kv_head, head_dim)
            # sin, cos = expand_sin_and_cos(q, sin, cos)
            q, k = self.apply_embedding(q, sin, cos, offset, k)
            v = self.apply_embedding(v, sin, cos, offset)
            return q, k, v
        else:
            raise NotImplementedError


class _IPEXRMSNormXPU(nn.Module):
    def __init__(self, module, config=None, tpp=False, woq=False):
        super().__init__()
        self.weight = module.weight
        self.eps = module.eps

    @classmethod
    def apply_function(cls, hidden_states, weight, eps):
        return torch.ops.torch_ipex.rms_norm(
            hidden_states, [hidden_states.size(-1)], weight, eps
        )[0]

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
    def apply_function(cls, hidden_states, normalized_shape, weight, bias, eps):
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
    def apply_function(
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
        softcap=-1.0,
    ):
        _IPEXVarlenScaledDotProductXPU.apply_function_flash_varlen(
            query,
            key,
            value,
            out,
            seqlen_q,
            seqlen_k,
            None,
            None,
            None,
            max_seqlen_q,
            max_seqlen_k,
            pdropout,
            softmax_scale,
            zero_tensors,
            is_causal,
            -1,
            -1,
            return_softmax,
            gen_,
            softcap,
        )
        return out

    @classmethod
    def apply_function_flash_varlen(
        cls,
        query,
        key,
        value,
        out,
        seqlen_q,
        seqlen_k,
        seqused_k,
        block_tables_,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        window_size_left,
        window_size_right,
        return_softmax,
        gen_,
        softcap=-1,
    ):
        assert (
            window_size_left == -1 and window_size_right == -1
        ), "IPEX only support window_size_left and window_size_right as -1"
        assert seqused_k is None, "IPEX only support seqused_k as None yet"
        assert block_tables_ is None, "IPEX only support block_tables_ as None yet"
        if torch.xpu.has_2d_block_array():
            torch.ops.torch_ipex.varlen_fwd(
                query,
                key,
                value,
                out,
                seqlen_q,
                seqlen_k,
                seqused_k,  # seqused_k
                alibi_slopes,  # alibi_slopes
                max_seqlen_q,
                max_seqlen_k,
                p_dropout,
                softmax_scale,
                zero_tensors,
                is_causal,
                return_softmax,
                gen_,
                softcap,
            )
        else:
            torch.xpu.varlen_fwd(
                query,
                key,
                value,
                out,
                seqlen_q,
                seqlen_k,
                max_seqlen_q,
                max_seqlen_k,
                p_dropout,
                softmax_scale,
                zero_tensors,
                is_causal,
                return_softmax,
                gen_,
                softcap,
            )
        return out

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
        softcap=-1.0,
    ):
        return self.apply_function(
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
            softcap,
        )


class _IPEXPagedAttentionXPU:
    PARTITION_SIZE = 512

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
        query = query.contiguous()
        torch.ops.torch_ipex.paged_attention(
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

    @classmethod
    def swap_blocks(cls, src, dst, block_mapping):
        assert isinstance(block_mapping, Dict) or isinstance(
            block_mapping, torch.Tensor
        ), "We only support block_mapping as dict or torch tensor"
        if isinstance(block_mapping, Dict):
            block_mapping_tensor = []
            for key, values in block_mapping.items():
                if hasattr(values, "__iter__"):
                    for value in values:
                        block_mapping_tensor.append([key, value])
                else:
                    block_mapping_tensor.append([key, value])
            block_mapping = torch.tensor(
                block_mapping_tensor, device="xpu", dtype=torch.int64
            )
        return torch.ops.torch_ipex.swap_blocks(src, dst, block_mapping)

    @classmethod
    def copy_blocks(cls, key_caches, value_caches, block_mapping):
        assert isinstance(block_mapping, Dict) or isinstance(
            block_mapping, torch.Tensor
        ), "We only support block_mapping as dict or torch tensor"
        if isinstance(block_mapping, Dict):
            block_mapping_tensor = []
            for key, values in block_mapping.items():
                if hasattr(values, "__iter__"):
                    for value in values:
                        block_mapping_tensor.append([key, value])
                else:
                    block_mapping_tensor.append([key, value])
            block_mapping = torch.tensor(
                block_mapping_tensor, device="xpu", dtype=torch.int64
            )
        return torch.ops.torch_ipex.copy_blocks(key_caches, value_caches, block_mapping)
