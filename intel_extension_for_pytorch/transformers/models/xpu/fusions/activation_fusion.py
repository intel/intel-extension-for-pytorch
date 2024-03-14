import torch


def silu_mul_xpu(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor = None):
    if out is None:
        out = torch.empty_like(x)
    torch.ops.torch_ipex.silu_mul(x, y, out)
    return out


def gelu_mul_xpu(
    x: torch.Tensor,
    y: torch.Tensor,
    out: torch.Tensor = None,
    approximate: str = "none",
):
    if out is None:
        out = torch.empty_like(x)
    torch.ops.torch_ipex.gelu_mul(x, y, out, approximate)
    return out


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
