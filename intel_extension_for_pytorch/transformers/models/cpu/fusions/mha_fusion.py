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
        kwargs=None,
    ):
        super().__init__()
        self.embed_positions = RotaryEmbedding(
            max_position_embeddings, pos_embd_dim, backbone, base, kwargs
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
            # query, key (in/out shape) : [bs, seqlen, num_head/num_kv_head, head_dim]
            # sin, cos: [seqlen, rotary_dim]
            # position_ids: [bs, seqlen]
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
            # concat_qkv (in shape) : [bs, seqlen, hidden_size*3]
            # query, key, value (out shape) : [bs, seqlen, num_head/num_kv_head, head_dim]
            # sin, cos: [seqlen, rotary_dim]
            # position_ids: [bs, seqlen]
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

    @classmethod
    def rotary_embedding(
        cls, query, key, sin, cos, rotary_dim, rotary_half, position_ids=None
    ):
        # query, key (in/out shape) torch.Tensor :
        #    4D: [bs, seqlen, num_head/num_kv_head, head_dim]
        #    3D: [num_tokens, num_head/num_kv_head, head_dim]
        # sin, cos: torch.Tensor [num_tokens, rotary_dim]
        # position_ids (optional): torch.Tensor [bs, seqlen]
        head_dim = query.size(-1)
        num_head = query.size(-2)
        num_kv_head = key.size(-2)
        input_3d = False
        assert (
            key.dim() == query.dim() and query.dim() == 3 or query.dim() == 4
        ), "rotary embedding query/key dim == 3 or 4"

        if query.dim() == 3:
            input_3d = True
            query_ = query.unsqueeze(0)
            key_ = key.unsqueeze(0)
        else:
            query_ = query
            key_ = key

        if rotary_half:
            offset = rotary_dim // 2
        else:
            offset = 1

        if position_ids is None:
            position_ids = torch.tensor(0).to(torch.long)

        assert (sin.size(-1) * 2 == head_dim or sin.size(-1) == head_dim) and sin.size(
            -1
        ) == cos.size(
            -1
        ), "rotary embedding only support sin and cos's last dim equals query's last dim or half of last dim"

        if sin.size(-1) == head_dim:
            _sin, _ = torch.split(sin, sin.shape[-1] // 2, dim=-1)
            _cos, _ = torch.split(cos, cos.shape[-1] // 2, dim=-1)
        else:
            _sin = sin
            _cos = cos

        sin_cos = (
            torch.cat((_sin.view(-1, _sin.shape[-1]), _cos.view(-1, _cos.shape[-1])), 1)
            .contiguous()
            .to(torch.float32)
            .view(-1, head_dim)
        )

        query_, _, _ = torch.ops.torch_ipex.rotary_position_embedding(
            query_,
            sin_cos,
            position_ids,
            num_head,
            head_dim,
            offset,
            rotary_dim,
        )

        key_, _, _ = torch.ops.torch_ipex.rotary_position_embedding(
            key_,
            sin_cos,
            position_ids,
            num_kv_head,
            head_dim,
            offset,
            rotary_dim,
        )
        if input_3d:
            query_ = query_.view([-1, num_head, head_dim])
            key_ = key_.view([-1, num_kv_head, head_dim])
        # keep the inplace context as used in TGI
        query.copy_(query_)
        key.copy_(key_)
        return query, key


class _IPEXScaleDotProductCPU(nn.Module):
    def __init__(self, text_max_length):
        super().__init__()
        self.text_max_length = text_max_length

    @classmethod
    def apply_function(
        cls,
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
        text_max_length: Optional[int] = 0,
        cutoff: Optional[torch.Tensor] = None,
        vision: Optional[torch.Tensor] = False,
        cache_type: Optional[torch.dtype] = None,
    ):
        if layer_past is None and cache_type is None:
            cache_type = key.dtype
        if cutoff is not None:
            if layer_past is None:
                layer_past = (
                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                    torch.zeros(
                        [key.size(0), key.size(2), 1, key.size(-1)],
                        dtype=cache_type,
                    ).contiguous(),
                    torch.zeros(
                        [value.size(0), value.size(2), 1, value.size(-1)],
                        dtype=cache_type,
                    ).contiguous(),
                    torch.zeros(1, int(query.size(0)), dtype=torch.long).contiguous(),
                )
            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)
            kc = torch.cat([layer_past[1].to(query.dtype), key[:, :, -1:, :]], dim=2)
            vc = torch.cat([layer_past[2].to(query.dtype), value[:, :, -1:, :]], dim=2)
            key = torch.cat([key[:, :, :cutoff, :], kc[:, :, 1:, :]], dim=2)
            value = torch.cat(
                [value[:, :, :cutoff, :], vc[:, :, 1:, :]],
                dim=2,
            )
            attn_output, _ = torch.ops.torch_ipex.flash_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                attention_mask=attention_mask,
            )
            present = (
                torch.empty(
                    1,
                    (layer_past[0].size(-2) + 1),
                    (layer_past[0].size(-2) + 1),
                    1,
                    dtype=torch.long,
                ).contiguous(),
                key[:, :, cutoff - 1 :, :],
                value[:, :, cutoff - 1 :, :],
                layer_past[3].contiguous(),
            )
            return attn_output, None, present
        if attention_mask is None:
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()
            value = value.transpose(1, 2).contiguous()
            attn_output, _ = torch.ops.torch_ipex.flash_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                attention_mask=attention_mask,
            )
            return attn_output, None, None
        if layer_past is None:
            layer_past = (
                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                torch.zeros([1, 1, 1, 1]).contiguous().to(cache_type),
                torch.zeros([1, 1, 1, 1]).contiguous().to(cache_type),
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
            text_max_length,
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
        cutoff: Optional[torch.Tensor] = None,
        vision: Optional[torch.Tensor] = False,
        cache_type: Optional[torch.dtype] = None,
    ):
        return self.apply_function(
            query,
            key,
            value,
            scale_attn,
            layer_past,
            head_mask,
            attention_mask,
            alibi,
            add_casual_mask,
            seq_info,
            self.text_max_length,
            cutoff,
            vision,
            cache_type,
        )


class _IPEXRMSNormCPU(nn.Module):
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

    @classmethod
    def apply_function(cls, hidden_states, weight, eps):
        return torch.ops.torch_ipex.rmsnorm(hidden_states, weight=weight, eps=eps)


class _IPEXFastLayerNormCPU(nn.Module):
    def __init__(self, normalized_shape, eps, weight, bias=None):
        super().__init__()
        self.module = nn.LayerNorm(normalized_shape, eps)
        self.module.weight = weight
        self.module.bias = bias

    def forward(self, hidden_states):
        return self.module(hidden_states)

    @classmethod
    def apply_function(cls, hidden_states, normalized_shape, weight, bias, eps):
        return torch.nn.functional.layer_norm(
            hidden_states, normalized_shape, weight=weight, bias=bias, eps=eps
        )


class _IPEXPagedAttentionCPU:
    @classmethod
    def reshape_and_cache(
        cls, key, value, key_cache, value_cache, slot_mapping, k_scale=1.0, v_scale=1.0
    ):
        torch.ops.torch_ipex.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.int() if slot_mapping.dtype is torch.long else slot_mapping,
            k_scale,
            v_scale,
        )

    @classmethod
    def reshape_and_cache_flash(
        cls, key, value, key_cache, value_cache, slot_mapping, k_scale=1.0, v_scale=1.0
    ):
        torch.ops.torch_ipex.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.int() if slot_mapping.dtype is torch.long else slot_mapping,
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
        window_size=-1,
        k_scale=1.0,
        v_scale=1.0,
        softcap=-1.0,
    ):
        torch.ops.torch_ipex.single_query_cached_kv_attention(
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
            window_size,
            k_scale,
            v_scale,
            softcap,
        )

    @classmethod
    def flash_attn_varlen_func(
        cls,
        output,
        query,
        k_cache,
        v_cache,
        cu_seq_lens_q,
        cu_seq_lens_kv,
        max_seq_len_q,
        max_seq_len_kv,
        scale,
        is_causal,
        block_table,
        alibi_slopes=None,
        window_size_left=-1,
        window_size_right=-1,
        k_scale=1.0,
        v_scale=1.0,
        softcap=-1.0,
    ):
        torch.ops.torch_ipex.flash_attn_varlen_func(
            output,
            query,
            k_cache,
            v_cache,
            cu_seq_lens_q,
            cu_seq_lens_kv,
            max_seq_len_q,
            max_seq_len_kv,
            scale,
            is_causal,
            block_table,
            alibi_slopes,
            window_size_left,
            window_size_right,
            k_scale,
            v_scale,
            softcap,
        )


class _IPEXVarlenScaledDotProductCPU(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def repeat_kv(cls, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        return torch.repeat_interleave(x, dim=1, repeats=n_rep)

    @classmethod
    def apply_function(
        cls,
        query,  # [total_q, num_head, head_size]
        key,  # [total_k, num_head_k, head_size]
        value,  # [total_k, num_head_k, head_size]
        out,  # [total_q, num_head, head_size]
        seqlen_q,  # [batch_size + 1]
        seqlen_k,  # [batch_size + 1]
        max_seqlen_q,
        max_seqlen_k,
        pdropout=0.0,
        softmax_scale=None,
        zero_tensors=False,
        is_causal=True,
        return_softmax=False,
        gen_=None,
    ):
        assert return_softmax is False, "ipex do not support return_softmax option"
        assert gen_ is None, "ipex do not support custom random generator"
        assert zero_tensors is False, "ipex varlen_fwd do not support zero tensors"

        # Repeat kv if it is GQA.
        key = cls.repeat_kv(key, int(query.shape[1] / key.shape[1]))
        value = cls.repeat_kv(value, int(query.shape[1] / value.shape[1]))

        total_q, num_head, head_size = query.size()
        total_k, num_head_k, _ = key.size()
        batch_size = seqlen_q.size(0) - 1
        seqlen_q_ = seqlen_q.clone()
        seqlen_q_[:batch_size] = seqlen_q[1:]
        seqlen_q = (seqlen_q_ - seqlen_q)[:batch_size]
        seqlen_k_ = seqlen_k.clone()
        seqlen_k_[:batch_size] = seqlen_k[1:]
        seqlen_k = (seqlen_k_ - seqlen_k)[:batch_size]

        pad_q = torch.zeros(
            [batch_size, max_seqlen_q, num_head, head_size],
            dtype=query.dtype,
            device=query.device,
        )
        pad_k = torch.zeros(
            [batch_size, max_seqlen_k, num_head_k, head_size],
            dtype=key.dtype,
            device=key.device,
        )
        pad_v = torch.zeros(
            [batch_size, max_seqlen_k, num_head_k, head_size],
            dtype=value.dtype,
            device=value.device,
        )
        q_mask = torch.arange(0, max_seqlen_q, device=query.device)[None, :].repeat(
            batch_size, 1
        )
        q_mask = q_mask < seqlen_q[:, None].repeat(1, q_mask.size(-1))
        k_mask = torch.arange(0, max_seqlen_k, device=key.device)[None, :].repeat(
            batch_size, 1
        )
        k_mask = k_mask < seqlen_k[:, None].repeat(1, k_mask.size(-1))
        align_mask_seqlen = max_seqlen_k
        attn_mask = torch.empty(
            [batch_size, 1, 1, align_mask_seqlen],
            dtype=query.dtype,
            device=query.device,
        ).fill_(float("-inf"))
        attn_mask[:, :, :, :max_seqlen_k].masked_fill_(k_mask[:, None, None, :], 0)

        pad_q[q_mask] = query
        pad_k[k_mask] = key
        pad_v[k_mask] = value

        pad_q = pad_q.permute(0, 2, 1, 3)
        pad_k = pad_k.permute(0, 2, 1, 3)
        pad_v = pad_v.permute(0, 2, 1, 3)
        out_ = torch.nn.functional.scaled_dot_product_attention(
            pad_q,
            pad_k,
            pad_v,
            attn_mask=attn_mask if not is_causal else None,
            is_causal=is_causal,
        )
        out_ = out_.permute(0, 2, 1, 3)
        out.copy_(out_[q_mask])
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
        )


class _IPEXMambaMixerCPU:
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
        out, final_states_out = torch.ops.torch_ipex.causal_conv1d_fn(
            x, weight, bias, initial_states, final_states_out, activation == "silu"
        )
        return (out, None) if not return_final_states else (out, final_states_out)

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
        out, conv_state_new = torch.ops.torch_ipex.causal_conv1d_update(
            x, conv_state, weight, bias, activation == "silu", cache_seqlens
        )
        conv_state.copy_(conv_state_new)
        return out

    @classmethod
    def selective_state_update(
        cls, state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
    ):
        """
        Argument:
            state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
            x: (batch, dim) or (batch, nheads, dim)
            dt: (batch, dim) or (batch, nheads, dim)
            A: (dim, dstate) or (nheads, dim, dstate) or (dstate, dim) or (nheads, dstate, dim)
            B: (batch, dstate) or (batch, ngroups, dstate)
            C: (batch, dstate) or (batch, ngroups, dstate)
            D: (dim,) or (nheads, dim) or None
            z: (batch, dim) or (batch, nheads, dim) or None
            dt_bias: (dim,) or (nheads, dim) or None
            dt_softplus: bool
        Return:
            out: (batch, dim) or (batch, nheads, dim)
        """
        return torch.ops.torch_ipex.selective_state_update(
            state, x, dt, A, B, C, D, z, dt_bias, dt_softplus
        )

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
        out, ssm_state = torch.ops.torch_ipex.selective_scan_fn(
            u,
            delta.to(A.dtype),
            A,
            B,
            C,
            D,
            z,
            delta_bias,
            delta_softplus,
            return_last_state,
        )
        return out if not return_last_state else (out, ssm_state)


def add_rms_norm_cpu(
    add: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    add_back: bool,
):
    assert bias is None, "bias is not supported in add_rmsnorm yet"
    if add is not None:
        return torch.ops.torch_ipex.add_rmsnorm(x, add, weight, eps, add_back)
    else:
        return torch.ops.torch_ipex.rmsnorm(x, weight, eps)


def add_layer_norm_cpu(
    add: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    add_back: bool,
):
    if add is not None:
        out = torch.ops.torch_ipex.add_layernorm(
            x, add, 1, [x.size(-1)], weight, bias, eps
        )
        if add_back:
            add.add_(x)
        return out
    else:
        return torch.nn.functional.layer_norm(
            x, [x.size(-1)], weight=weight, bias=bias, eps=eps
        )


@torch.compile(dynamic=True, options={"fx_graph_cache": True})
def silu_mul_cpu(x, y, out=None):
    res = torch.nn.functional.silu(x) * y
    if out is not None:
        out.copy_(res)
    return res


@torch.compile(dynamic=True, options={"fx_graph_cache": True})
def gelu_mul_cpu(x, y, out=None, approximate="none"):
    res = torch.nn.functional.gelu(x, approximate=approximate) * y
    if out is not None:
        out.copy_(res)
    return res


def bgmv_shrink_cpu(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float,
):
    r"""
    Args:
        inputs (torch.Tensor): The input tensor with shape of [batchsize, hidden_size].
        lora_a_weights (torch.Tensor):  LoRA weights tensor with shape of [num_lora, max_rank, hidden_size].
        output_tensor (torch.Tensor): The output tensor with shape of [batchsize, output_size1] which output_size1 >= max_rank
        lora_indices_tensor (torch.Tensor): The tensor mapping each input token to
          the lora-id related to that token with shape of [batchsize].
        scaling (float): Scaling factor.
    """
    torch.ops.torch_ipex.punica_bgmv_shrink(
        output_tensor, inputs, lora_a_weights, lora_indices_tensor, scaling
    )
    return


def bgmv_expand_cpu(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool,
):
    r"""
    Args:
        inputs (torch.Tensor): The input tensor with shape of
            [batchsize, input_size1] or [1, input_size1] which input_size1  >= hidden_size.
        lora_b_weights (torch.Tensor):  LoRA weights tensor
            with shape of [num_lora, max_rank, hidden_size].
        output_tensor (torch.Tensor): The output tensor with shape of
            [batchsize, output_size1] which output_size1 >= max_rank
        lora_indices_tensor (torch.Tensor): The tensor mapping each input token
            to the lora-id related to that token with shape of [batchsize].
        add_inputs (bool): Whether to add to the output tensor.
    """
    torch.ops.torch_ipex.punica_bgmv_expand(
        output_tensor, inputs, lora_b_weights, lora_indices_tensor, add_inputs
    )
    return


def bgmv_expand_slice_cpu(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool,
):
    r"""
    Args:
        inputs (torch.Tensor): The input tensor with shape of [batchsize, max_rank].
        lora_b_weights (torch.Tensor):  LoRA weights tensor with shape of [num_lora, hidden_size, max_rank].
        output_tensor (torch.Tensor): The output tensor with shape of [batchsize, output_size1]
            which output_size1 >= slice_offset + slice_size
        lora_indices_tensor (torch.Tensor): The tensor mapping each input token to
            the lora-id related to that token with shape of [batchsize].
        slice_offset (int): Slice offset start for output.
        slice_size (int): Slice length for output.
        add_inputs (bool): Whether to add to the output tensor.
    """
    torch.ops.torch_ipex.punica_bgmv_expand_slice(
        output_tensor,
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        slice_offset,
        slice_size,
        add_inputs,
    )
    return
