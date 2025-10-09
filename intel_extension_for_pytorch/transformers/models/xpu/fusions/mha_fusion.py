import torch
from torch import nn
from ...reference.fusions.mha_fusion import RotaryEmbedding
from typing import Optional, Dict
import os


def convert_from_fp8(
    q_dtype, key_cache, value_cache, k_scale=1.0, v_scale=1.0, kv_dtype="auto"
):
    """
    Convert key_cache and value_cache from FP8 or uint8 to the specified q_dtype.

    Args:
        q_dtype: Target data type for conversion.
        key_cache: Key cache tensor.
        value_cache: Value cache tensor.
        k_scale: Scaling factor for key cache.
        v_scale: Scaling factor for value cache.
        kv_dtype: Data type of key/value cache if uint8.

    Returns:
        Converted key_cache and value_cache tensors.
    """
    if key_cache.dtype != q_dtype:
        if key_cache.dtype == torch.float8_e4m3fn:
            key_cache = key_cache.to(q_dtype) * k_scale
            value_cache = value_cache.to(q_dtype) * v_scale
        elif key_cache.dtype == torch.float8_e5m2:
            key_cache = key_cache.to(q_dtype)
            value_cache = value_cache.to(q_dtype)
        elif key_cache.dtype == torch.uint8:
            if kv_dtype in ["float8_e4m3fn", "fp8"]:
                key_cache_fp8 = torch.empty_like(key_cache, dtype=torch.float8_e4m3fn)
                value_cache_fp8 = torch.empty_like(
                    value_cache, dtype=torch.float8_e4m3fn
                )
            else:
                key_cache_fp8 = torch.empty_like(key_cache, dtype=torch.float8_e5m2)
                value_cache_fp8 = torch.empty_like(value_cache, dtype=torch.float8_e5m2)

            key_cache_fp8.copy_(key_cache)
            value_cache_fp8.copy_(value_cache)
            key_cache = key_cache_fp8.to(q_dtype) * k_scale
            value_cache = value_cache_fp8.to(q_dtype) * v_scale

    return key_cache, value_cache


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
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        pdropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        return_softmax,
        gen_,
        window_size_left,
        window_size_right,
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
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            pdropout,
            softmax_scale,
            zero_tensors,
            is_causal,
            window_size_left,
            window_size_right,
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
        assert seqused_k is None, "IPEX only support seqused_k as None yet"
        assert block_tables_ is None, "IPEX only support block_tables_ as None yet"
        if torch.xpu.has_2d_block_array():
            head_dim = query.size(-1)
            pad_query = query
            pad_key = key
            pad_value = value
            pad_out = out
            if head_dim % 32 != 0:
                pad_size = 32 - head_dim % 32
                pad_query = torch.nn.functional.pad(query, (0, pad_size))
                pad_key = torch.nn.functional.pad(key, (0, pad_size))
                pad_value = torch.nn.functional.pad(value, (0, pad_size))
                pad_out = torch.nn.functional.pad(out, (0, pad_size))
            torch.ops.torch_ipex.varlen_fwd(
                pad_query,
                pad_key,
                pad_value,
                pad_out,
                seqlen_q,
                seqlen_k,
                seqused_k,  # seqused_k
                alibi_slopes,  # alibi_slopes
                max_seqlen_q,
                max_seqlen_k,
                p_dropout,
                softmax_scale,
                zero_tensors,
                window_size_left,
                window_size_right,
                is_causal,
                return_softmax,
                gen_,
                softcap,
            )
            if head_dim % 32 != 0:
                out.copy_(pad_out[:, :, :head_dim])
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
    def reshape_and_cache(
        cls,
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype="auto",
        k_scale=1.0,
        v_scale=1.0,
    ):  # todo, k_scale and v_scale not implement here. tmply add arguments for frontend alignment with CPU
        return torch.ops.torch_ipex.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    @classmethod
    def reshape_and_cache_flash(
        cls,
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype="auto",
        k_scale=1.0,
        v_scale=1.0,
    ):
        return torch.ops.torch_ipex.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
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
        ky_dtype="auto",
        window_size=-1,
        k_scale=1.0,
        v_scale=1.0,
        softcap=-1.0,
    ):
        key_cache, value_cache = convert_from_fp8(
            query.dtype, key_cache, value_cache, k_scale, v_scale, ky_dtype
        )
        num_queries_per_tokens = (head_mapping == 0).sum()
        query = query.contiguous()
        return torch.ops.torch_ipex.paged_attention(
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            num_queries_per_tokens,
            scale,
            block_size,
            max_context_len,
            alibi_slopes,
            softcap,
        )

    @classmethod
    def single_query_kv_attention(
        cls,
        output,
        query,
        key_cache,
        value_cache,
        num_queries_per_tokens,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        softcap=-1.0,
    ):
        query = query.contiguous()
        key_cache, value_cache = convert_from_fp8(query.dtype, key_cache, value_cache)
        return torch.ops.torch_ipex.paged_attention(
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            num_queries_per_tokens,
            scale,
            block_size,
            max_context_len,
            alibi_slopes,
            softcap,
        )

    @classmethod
    def flash_attn_varlen_func(
        cls,
        output,
        query,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        scale,
        is_causal,
        block_table,
        alibi_slopes=None,
        sink=None,
        window_size_left=-1,
        window_size_right=-1,
        kv_cache_dtype: str = "auto",
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        softcap: float = -1.0,
    ):
        k_cache, v_cache = convert_from_fp8(
            query.dtype, k_cache, v_cache, k_scale, v_scale, kv_cache_dtype
        )
        head_dim = query.size(-1)
        pad_query = query
        pad_k_cache = k_cache
        pad_v_cache = v_cache
        block_table_s = block_table
        pad_output = output
        disable_unique = os.environ.get(
            "DISABLE_UNIQUE_FOR_CHUNKED_PREFILL", "OFF"
        ).upper() in ["1", "Y", "ON", "YES", "TRUE"]
        if head_dim % 64 != 0 and not disable_unique:
            pad_size = 64 - head_dim % 64
            pad_query = torch.nn.functional.pad(query, (0, pad_size))
            block_valid, block_table_s = block_table.unique(return_inverse=True)
            pad_k_cache = torch.nn.functional.pad(k_cache[block_valid], (0, pad_size))
            pad_v_cache = torch.nn.functional.pad(v_cache[block_valid], (0, pad_size))
            pad_output = torch.nn.functional.pad(output, (0, pad_size))
            block_table_s = block_table_s.to(torch.int32)
        elif head_dim % 64 != 0 and disable_unique:
            pad_size = 64 - head_dim % 64
            pad_query = torch.nn.functional.pad(query, (0, pad_size))
            pad_k_cache = torch.nn.functional.pad(k_cache, (0, pad_size))
            pad_v_cache = torch.nn.functional.pad(v_cache, (0, pad_size))
            pad_output = torch.nn.functional.pad(output, (0, pad_size))
        block_size = pad_k_cache.size(1)
        if block_size * block_table_s.size(1) > max_seqlen_kv:
            max_block_per_seq = (max_seqlen_kv + block_size - 1) // block_size
            block_table_s = block_table_s[:, :max_block_per_seq].contiguous()
        torch.ops.torch_ipex.chunked_prefill(
            pad_query,
            pad_k_cache,
            pad_v_cache,
            pad_output,
            cu_seqlens_q,
            cu_seqlens_kv,
            None,
            block_table_s,
            alibi_slopes,
            sink,
            max_seqlen_q,
            max_seqlen_kv,
            0.0,
            scale,
            False,
            window_size_left,
            window_size_right,
            is_causal,
            False,
            None,
            softcap,
        )
        if head_dim % 64 != 0:
            output.copy_(pad_output[:, :, :head_dim])
        return output

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


class _IPEXMambaMixerXPU:
    @classmethod
    def causal_conv1d_fn(
        cls,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        initial_states: Optional[torch.Tensor] = None,
        return_final_states: bool = False,
        final_states_out: Optional[torch.Tensor] = None,
        activation: Optional[str] = "silu",
    ):
        """
        x: (batch, dim, seqlen)
        weight: (dim, width)
        bias: (dim,)
        initial_states: (batch, dim, width - 1)
        final_states_out: (batch, dim, width - 1)

        out: (batch, dim, seqlen)
        """
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")
        if x.stride(-1) != 1:
            x = x.contiguous()
        bias = bias.contiguous() if bias is not None else None
        torch.ops.torch_ipex.causal_conv1d_fn(
            x,
            weight,
            bias,
            initial_states,
            None,
            None,
            None,
            activation in ["silu", "swish"],
            -1,
        )
        return (x, None) if not return_final_states else (x, initial_states)

    @classmethod
    def causal_conv1d_update(
        cls, x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
    ):
        """
        x: (batch, dim) or (batch, dim, seqlen)
        conv_state: (batch, dim, state_len), where state_len >= width - 1
        weight: (dim, width)
        bias: (dim,)
        activation: None, "silu", or "swish"
        cache_seqlens: (batch,), dtype int32.
            If not None, the conv_state is treated as a circular buffer.
            The conv_state will be updated by copying x to the
            conv_state starting at the index
            @cache_seqlens % state_len before performing the convolution.

        out: (batch, dim) or (batch, dim, seqlen)
        """
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")
        activation_val = activation in ["silu", "swish"]
        unsqueeze = x.dim() == 2
        if unsqueeze:
            x = x.unsqueeze(-1)
        torch.ops.torch_ipex.causal_conv1d_update(
            x, conv_state, weight, bias, activation_val, cache_seqlens, None, -1
        )
        if unsqueeze:
            x = x.squeeze(-1)
        return x

    @classmethod
    def selective_scan_fn(
        cls,
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
    ):
        """
        u: (B D L) or (B L D)
        delta: same shape as u
        A: (D N) or (N D)
        B: (B N L) or (B N 2L) or (B G N L)
        C: (B N L) or (B N 2L) or (B G N L)
        D: (D) or None
        z: (B D L) or None
        delta_bias: (D) or None, fp32

        out: (B D L)
        last_state (optional): (B D dstate)
        """
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(1)
        if C.dim() == 3:
            C = C.unsqueeze(1)

        batch = u.shape[0]
        dim = u.shape[1]
        dstate = A.shape[1]
        ssm_state = torch.empty((batch, dim, dstate), dtype=u.dtype, device=u.device)
        torch.ops.torch_ipex.selective_scan_fn(
            u,
            delta,
            A,
            B,
            C,
            D,
            z,
            delta_bias,
            delta_softplus,
            None,
            None,
            None,
            ssm_state,
            -1,
        )
        if z is None:
            return (
                delta if not return_last_state else (delta, ssm_state)
            )  # output written inplace to delta
        else:
            return (
                z if not return_last_state else (z, ssm_state)
            )  # output written inplace to z
