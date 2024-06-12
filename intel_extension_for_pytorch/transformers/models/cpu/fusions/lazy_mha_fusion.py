import torch


@torch.compile(dynamic=True, options={"fx_graph_cache": True})
def lazy_silu_mul_cpu(x, y, out=None):
    res = torch.nn.functional.silu(x) * y
    if out is not None:
        out.copy_(res)
    return res


@torch.compile(dynamic=True, options={"fx_graph_cache": True})
def lazy_gelu_mul_cpu(x, y, out=None, approximate="none"):
    res = torch.nn.functional.gelu(x, approximate=approximate) * y
    if out is not None:
        out.copy_(res)
    return res
