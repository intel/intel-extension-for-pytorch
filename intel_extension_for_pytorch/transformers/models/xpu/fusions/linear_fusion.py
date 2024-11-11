import torch
from torch import nn


class _IPEXlinearFusionXPU(nn.Module):
    def __init__(self, linear, tpp=False, woq=False):
        super().__init__()
        # Do not support tpp & woq for now
        self.tpp = tpp
        self.woq = woq
        self.dtype = None if woq else linear.weight.dtype

    def extra_repr(self):
        extra_repr_str = f"dtype = {self.dtype}, tpp = {self.tpp}, woq = {self.woq}"
        return extra_repr_str


class _IPEXlinearAddXPU(_IPEXlinearFusionXPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.weight = module.weight.transpose(0, 1).contiguous()
        self.bias = module.bias

    def forward(self, x, y):
        if self.bias is not None:
            x = torch.ops.torch_ipex.mm_bias_resadd(
                x, self.weight, self.bias, 1.0, y, 1.0
            )
        else:
            x = torch.addmm(
                y.flatten(0, -2),
                x.flatten(0, -2),
                self.weight,
                beta=1.0,
            )
        return x


class _IPEXlinearAddAddXPU(_IPEXlinearFusionXPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.weight = module.weight.transpose(0, 1).contiguous()
        self.bias = module.bias

    def forward(self, x, y, z):
        if self.bias is not None:
            x = torch.ops.torch_ipex.mm_bias_resadd(
                x, self.weight, self.bias, 1.0, y, 1.0
            )
            x += z
        else:
            x = torch.ops.torch_ipex.mm_bias_resadd(x, self.weight, z, 1.0, y, 1.0)
        return x


class _IPEXlinearGeluXPU(_IPEXlinearFusionXPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.weight = module.weight.transpose(0, 1).contiguous()
        self.bias = module.bias

    def forward(self, x):
        return torch.ops.torch_ipex.matmul_gelu(x, self.weight, self.bias, 1.0, "tanh")


class _IPEXlinearSiluMulXPU(_IPEXlinearFusionXPU):
    def __init__(self, module_1, module_2, tpp=False, woq=False):
        super().__init__(module_1, tpp=tpp, woq=woq)
        self.weight_1 = module_1.weight.transpose(0, 1).contiguous()
        self.weight_2 = module_2.weight.transpose(0, 1).contiguous()
        self.bias_1 = module_1.bias
        self.bias_2 = module_2.bias

    def forward(self, x):
        if self.bias_1 is not None:
            silu = nn.SiLU()
            linear_1_bias = torch.ops.torch_ipex.mm_resadd(
                x, self.weight_1, self.bias_1.unsqueeze(0), 1.0
            )
            silu_mm = silu(linear_1_bias)
        else:
            silu_mm = torch.ops.torch_ipex.matmul_silu(x, self.weight_1)

        if self.bias_2 is not None:
            linear_2_bias = torch.ops.torch_ipex.mm_resadd(
                x, self.weight_2, self.bias_2.unsqueeze(0), 1.0
            )
            return silu_mm * linear_2_bias
        else:
            return torch.ops.torch_ipex.mm_resmul(x, self.weight_2, silu_mm)
