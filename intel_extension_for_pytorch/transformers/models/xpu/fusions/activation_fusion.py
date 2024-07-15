import torch
from typing import Optional


def gelu_quick_xpu(x: torch.Tensor, out: Optional[torch.Tensor] = None):
    if out is None:
        out = torch.empty_like(x)
    return torch.ops.torch_ipex.gelu_quick_out(x, out)


def silu_mul_xpu(x: torch.Tensor, y: torch.Tensor, out: Optional[torch.Tensor] = None):
    if out is None:
        out = torch.empty_like(x)
    torch.ops.torch_ipex.silu_mul(x, y, out)
    return out


def silu_and_mul_xpu(x: torch.Tensor, out: Optional[torch.Tensor] = None):
    if out is None:
        d = x.size(-1) / 2
        out_shape = x.shape[:-1] + (d,)
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return torch.ops.torch_ipex.silu_and_mul(x, out)


def gelu_mul_xpu(
    x: torch.Tensor,
    y: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    approximate: str = "none",
):
    if out is None:
        out = torch.empty_like(x)
    torch.ops.torch_ipex.gelu_mul(x, y, out, approximate)
    return out


def gelu_and_mul_xpu(
    x: torch.Tensor, out: Optional[torch.Tensor] = None, approximate: str = "none"
):
    if out is None:
        d = x.size(-1) / 2
        out_shape = x.shape[:-1] + (d,)
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return torch.ops.torch_ipex.gelu_and_mul(x, out, approximate)


def add_rms_norm_xpu(
    add: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    add_back: bool,
):
    out = torch.ops.torch_ipex.add_rms_norm(
        add, x, [x.size(-1)], weight, bias, eps, add_back
    )
    return out


def add_layer_norm_xpu(
    add: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    add_back: bool,
):
    out = torch.ops.torch_ipex.add_layer_norm(
        add, x, [x.size(-1)], weight, bias, eps, add_back
    )
    return out


def rotary_embedding_batched_xpu(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_nexo_style: bool,
    rotary_dim: int,
    offsets: Optional[torch.Tensor] = None,
):
    if offsets is None:
        torch.ops.torch_ipex.rotary_embedding(
            positions, query, key, head_size, cos_sin_cache, is_nexo_style, rotary_dim
        )
    else:
        torch.ops.torch_ipex.rotary_embedding_batched(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_nexo_style,
            rotary_dim,
            offsets,
        )
    return query, key
