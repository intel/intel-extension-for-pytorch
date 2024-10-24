import torch
import intel_extension_for_pytorch as ipex  # noqa
from typing import Tuple, Optional
import pytest
import math

# The largest head dim we can support is 256
HEAD_DIM = [64, 128, 256]

NUM_HEADS = [(32, 32), (32, 8)]

BATCH_SIZE = [1, 3, 8]

DTYPE = [torch.float16]

USE_ALIBI = [False, True]

SEQLEN_RANGE = [10, 64, 500, 1024]

IS_CAUSAL = [False, True]


def varlen_fwd_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: Optional[torch.Tensor],
    seqlen_q: torch.Tensor,
    seqlen_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    alibi: Optional[torch.Tensor],
    pdropout: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    return_softmax: bool,
    gen_,
):
    assert return_softmax is False, "ipex do not support return_softmax option"
    assert gen_ is None, "ipex do not support custom random generator"
    assert zero_tensors is False, "ipex varlen_fwd do not support zero tensors"
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
    align_mask_seqlen = (max_seqlen_k + 63) // 64 * 64
    attn_mask = torch.empty(
        [batch_size, 1, 1, align_mask_seqlen], dtype=torch.float16, device=query.device
    ).fill_(float("-inf"))
    attn_mask[:, :, :, :max_seqlen_k].masked_fill_(k_mask[:, None, None, :], 0)
    # print("attn mask: ", attn_mask)
    pad_q[q_mask] = query
    pad_k[k_mask] = key
    pad_v[k_mask] = value
    block_alibi = None
    if alibi is not None:
        block_alibi = torch.empty(
            [batch_size, num_head, 1, align_mask_seqlen],
            device="xpu",
            dtype=query.dtype,
        )
        if alibi.dim() == 1:
            block_alibi[:, :, :, :max_seqlen_k] = alibi.view([1, num_head, 1, 1])
        elif alibi.dim() == 2:
            block_alibi[:, :, :, :max_seqlen_k] = alibi.view(
                [batch_size, num_head, 1, 1]
            )
        else:
            raise RuntimeError

    if num_head_k < num_head:
        assert num_head % num_head_k == 0, "num_head_k should be divisible by num_head."
        pad_k = pad_k.view([batch_size, max_seqlen_k, num_head_k, 1, head_size])
        pad_v = pad_v.view([batch_size, max_seqlen_k, num_head_k, 1, head_size])
        pad_k = pad_k.repeat([1, 1, 1, num_head // num_head_k, 1]).contiguous()
        pad_v = pad_v.repeat([1, 1, 1, num_head // num_head_k, 1]).contiguous()
        pad_k = pad_k.view([batch_size, max_seqlen_k, num_head, head_size])
        pad_v = pad_v.view([batch_size, max_seqlen_k, num_head, head_size])
    pad_q = pad_q.permute(0, 2, 1, 3)
    pad_k = pad_k.permute(0, 2, 1, 3)
    pad_v = pad_v.permute(0, 2, 1, 3)

    out_ = torch.ops.torch_ipex.xetla_fsdp_forward_atten_mask_alibi_strided(
        pad_q,
        pad_k,
        pad_v,
        block_alibi,
        attn_mask,
        None,
        softmax_scale,
        1.0,
        pdropout,
        is_causal,
        False,
    )
    q_mask = torch.arange(0, max_seqlen_q, device=query.device)[None, :].repeat(
        batch_size, 1
    )
    q_mask = q_mask < seqlen_q[:, None].repeat(1, q_mask.size(-1))
    out_ = out_.permute(0, 2, 1, 3).contiguous()
    out_cpu = out_.cpu()
    mask_cpu = q_mask.cpu()
    selected_out = out_cpu[mask_cpu]
    if out is None:
        return out_[q_mask]
    else:
        out.copy_(selected_out.to("xpu"))


# not support on DG2 yet
@pytest.mark.skipif(not torch.xpu.has_2d_block_array(), reason="fallback is required")
@pytest.mark.parametrize("head_dim", HEAD_DIM)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("seqlen_range", SEQLEN_RANGE)
@pytest.mark.parametrize("is_causal", IS_CAUSAL)
def test_varlen_fwd(
    head_dim: int,
    num_heads: Tuple[int, int],
    batch_size: int,
    dtype: torch.device,
    use_alibi: bool,
    seqlen_range: int,
    is_causal: bool,
):
    torch.manual_seed(15)
    seqlen_list = torch.randint(1, seqlen_range, [batch_size], dtype=torch.int32)
    max_seqlen = torch.max(seqlen_list)
    cu_seqlen = torch.cumsum(seqlen_list, dim=0)
    num_heads_query, num_heads_kv = num_heads
    cu_seqlen = (
        torch.cat([torch.tensor([0]), cu_seqlen], dim=0).to(torch.int32).to("xpu")
    )

    query = torch.randn(
        [cu_seqlen[-1], num_heads_query, head_dim], dtype=dtype, device="xpu"
    )
    key = torch.randn(
        [cu_seqlen[-1], num_heads_kv, head_dim], dtype=dtype, device="xpu"
    )
    value = torch.randn(
        [cu_seqlen[-1], num_heads_kv, head_dim], dtype=dtype, device="xpu"
    )
    alibi_slopes = None
    softmax_scale = 1 / math.sqrt(head_dim)
    if use_alibi:
        alibi_slopes = torch.randn(
            [batch_size, num_heads_query], dtype=dtype, device="xpu"
        )
    out_ref = query.clone()
    out = query.clone()
    ipex.llm.functional.varlen_fwd(
        query,
        key,
        value,
        out,
        cu_seqlen,
        cu_seqlen,
        None,
        None,
        alibi_slopes if use_alibi else None,
        max_seqlen,
        max_seqlen,
        0.0,
        softmax_scale,
        False,
        is_causal,
        -1,
        -1,
        False,
        None,
    )

    varlen_fwd_reference(
        query,
        key,
        value,
        out_ref,
        cu_seqlen,
        cu_seqlen,
        max_seqlen,
        max_seqlen,
        alibi_slopes,
        0.0,
        softmax_scale,
        False,
        is_causal,
        False,
        None,
    )
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=1e-3)

    ipex.llm.functional.varlen_attention(
        query,
        key,
        value,
        out,
        cu_seqlen,
        cu_seqlen,
        max_seqlen,
        max_seqlen,
        0.0,
        softmax_scale,
        False,
        is_causal,
        False,
        None,
    )
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=1e-3)


SOFTCAP = 50.0


@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.skipif(not torch.xpu.has_2d_block_array(), reason="fallback is required")
def test_varlen_attention_softcap(
    is_causal: bool,
):
    head_dim = 64
    num_heads = 32
    batch_size = 1
    seqlen_range = 10
    dtype = torch.float16

    torch.manual_seed(15)
    seqlen_list = torch.randint(1, seqlen_range, [batch_size], dtype=torch.int32)
    if not is_causal:
        seqlen_list = torch.ones(1, dtype=torch.int32)
    max_seqlen = torch.max(seqlen_list)
    cu_seqlen = torch.cumsum(seqlen_list, dim=0)
    num_heads_query, num_heads_kv = num_heads, num_heads
    cu_seqlen = (
        torch.cat([torch.tensor([0]), cu_seqlen], dim=0).to(torch.int32).to("xpu")
    )

    query = torch.randn(
        [cu_seqlen[-1], num_heads_query, head_dim], dtype=dtype, device="xpu"
    )
    key = torch.randn(
        [cu_seqlen[-1], num_heads_kv, head_dim], dtype=dtype, device="xpu"
    )
    value = torch.randn(
        [cu_seqlen[-1], num_heads_kv, head_dim], dtype=dtype, device="xpu"
    )
    out = query.clone()
    softmax_scale = 1 / math.sqrt(head_dim)
    ipex.llm.functional.varlen_attention(
        query,
        key,
        value,
        out,
        cu_seqlen,
        cu_seqlen,
        max_seqlen,
        max_seqlen,
        0.0,
        softmax_scale,
        False,
        is_causal,
        False,
        None,
        softcap=SOFTCAP,
    )
    query = query.transpose(0, 1).to(torch.float32)
    key = key.transpose(0, 1).to(torch.float32)
    value = value.transpose(0, 1).to(torch.float32)

    attn = query @ key.transpose(1, 2) * softmax_scale
    attn = torch.tanh(attn / SOFTCAP) * SOFTCAP
    if is_causal:
        causal_mask = torch.full(
            (query.shape[-2], query.shape[-2]),
            fill_value=torch.finfo(dtype).min,
            dtype=dtype,
            device="xpu",
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attn += causal_mask
    attn = torch.nn.functional.softmax(attn, dim=-1)
    out_ref = attn @ value
    out_ref = out_ref.transpose(0, 1).to(dtype)
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=1e-3)
