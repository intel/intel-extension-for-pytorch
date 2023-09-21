import torch
from torch import nn
import math


class _IPEXlinearSiluCPU(nn.Module):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__()
        self.linear = module
        self.tpp = tpp
        self.woq = woq
        self.dtype = self.linear.weight.dtype if self.tpp else None

    def forward(self, x):
        if self.tpp:
            x = x.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_linear_silu(
                x,
                self.linear.weight,
                self.linear.bias if self.linear.bias is not None else x.new_empty(0),
                self.linear.out_features,
            )
        else:  # fallback path
            return nn.functional.silu(self.linear(x))


class _IPEXlinearReluCPU(nn.Module):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__()
        self.linear = module
        self.tpp = tpp
        self.woq = woq
        self.dtype = self.linear.weight.dtype if self.tpp else None

    def forward(self, x):
        if self.tpp:
            x = x.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_linear_relu(
                x,
                self.linear.weight,
                self.linear.bias if self.linear.bias is not None else x.new_empty(0),
                self.linear.out_features,
            )
        else:  # fallback path
            return nn.functional.relu(self.linear(x))


class _IPEXlinearMulCPU(nn.Module):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__()
        self.linear = module
        self.tpp = tpp
        self.woq = woq
        self.dtype = self.linear.weight.dtype if self.tpp else None

    def forward(self, x, y):
        if self.tpp:
            x = x.to(self.dtype).contiguous()
            y = y.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_linear_mul(
                x,
                y,
                self.linear.weight,
                self.linear.bias if self.linear.bias is not None else x.new_empty(0),
                self.linear.out_features,
            )
        else:  # fallback path
            return self.linear(x) * y


class _IPEXlinearAddCPU(nn.Module):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__()
        self.linear = module
        self.tpp = tpp
        self.woq = woq
        self.dtype = self.linear.weight.dtype if self.tpp else None

    def forward(self, x, y):
        if self.tpp:
            x = x.to(self.dtype).contiguous()
            y = y.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_linear_add(
                x,
                y,
                self.linear.weight,
                self.linear.bias if self.linear.bias is not None else x.new_empty(0),
                1.0,
                self.linear.out_features,
            )
        else:  # fallback path
            return self.linear(x) + y


class _IPEXlinearAddAddCPU(nn.Module):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__()
        self.linear = module
        self.tpp = tpp
        self.woq = woq
        self.dtype = self.linear.weight.dtype if self.tpp else None

    def forward(self, x, y, z):
        if self.tpp:
            x = x.to(self.dtype).contiguous()
            y = y.to(self.dtype).contiguous()
            z = z.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_linear_add_add(
                x,
                y,
                z,
                self.linear.weight,
                self.linear.bias if self.linear.bias is not None else x.new_empty(0),
                1.0,
                self.linear.out_features,
            )
        else:  # fallback path
            return self.linear(x) + y + z


class _IPEXlinearNewGeluCPU(nn.Module):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__()
        self.linear = module
        self.tpp = tpp
        self.woq = woq
        self.dtype = self.linear.weight.dtype if self.tpp else None

    def forward(self, x):
        if self.tpp:
            x = x.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_linear_gelu(
                x,
                self.linear.weight,
                self.linear.bias if self.linear.bias is not None else x.new_empty(0),
                self.linear.out_features,
            )

        else:  # fallback path
            x = self.linear(x)
            return (
                0.5
                * x
                * (
                    1.0
                    + torch.tanh(
                        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                    )
                )
            )


class _IPEXlinearGeluCPU(nn.Module):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__()
        self.linear = module
        self.tpp = tpp
        self.woq = woq
        self.gelu = nn.GELU()
        self.dtype = self.linear.weight.dtype if self.tpp else None

    def forward(self, x):
        if self.tpp:
            x = x.to(self.dtype).contiguous()
            return torch.ops.torch_ipex.tpp_linear_gelu(
                x,
                self.linear.weight,
                self.linear.bias if self.linear.bias is not None else x.new_empty(0),
            )

        else:  # fallback path
            x = self.gelu(self.linear(x))
            return x
