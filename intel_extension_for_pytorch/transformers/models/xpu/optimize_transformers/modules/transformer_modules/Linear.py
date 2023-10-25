import torch
import torch.nn as nn


class IPEXTransformerLinear(nn.Module):
    def __init__(self, weight=None, bias=None) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input):
        return torch.ops.torch_ipex.matmul_bias_out(input, self.weight, self.bias)


class IPEXTransformerQLinear(nn.Module):
    def __init__(self, weight=None, bias=None, scale=None, zp=None, gs=None):
        super(IPEXTransformerQLinear, self).__init__()
        # we set the weight and bias to None to avoid any possible memory presure
        self.weight = None
        self.bias = None
        self.scale = None
        self.zp = None
        self.gs = None

    def forward(self, input):
        return torch.ops.torch_ipex.mm_int4(
            input, self.weight, self.scale, self.zp, self.gs
        )


def matmul_add_add(attn_output, weight, tp_size=1, bias=None, residual=None):
    seq_len, bs, _ = attn_output.size()
    if residual is None:
        attn_output = torch.matmul(attn_output, weight)
        if bias is not None:
            attn_output += bias
    else:
        if bias is not None:
            attn_output = torch.ops.torch_ipex.mm_bias_resadd(
                attn_output, weight, bias, 1.0 / tp_size, residual, 1.0 / tp_size
            )
        else:
            attn_output = torch.addmm(
                residual.flatten(0, -2),
                attn_output.flatten(0, -2),
                weight,
                beta=1.0 / tp_size,
            )
    attn_output = attn_output.view(seq_len, bs, -1)
    return attn_output
