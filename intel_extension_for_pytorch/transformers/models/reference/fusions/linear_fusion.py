import torch
from torch import nn
import math
import copy
from intel_extension_for_pytorch.nn.modules import WeightOnlyQuantizedLinear


class _IPEXlinearSiluRef(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.linear = module

    def forward(self, x):
        return nn.functional.silu(self.linear(x))


class _IPEXlinearAddRef(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.linear = module

    def forward(self, x, y):
        return self.linear(x) + y


class _IPEXlinearAddAddRef(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.linear = module

    def forward(self, x, y, z):
        return self.linear(x) + y + z


class _IPEXlinearMulRef(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.linear = module

    def forward(self, x, y):
        return self.linear(x) * y


class _IPEXlinearNewGeluRef(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.linear = module

    def forward(self, x):
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


class _IPEXlinearGeluRef(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.linear = module
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.linear(x))


class _IPEXlinearReluRef(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.linear = module

    def forward(self, x):
        return nn.functional.relu(self.linear(x))


class _IPEXConcatLinearRef(nn.Module):
    def __init__(self, linear_list: list):
        super().__init__()
        self.num_concat = len(linear_list)
        for i in range(self.num_concat):
            attr_name = f"linear_{i}"
            setattr(self, attr_name, copy.deepcopy(linear_list[i]))
        self.concat_linear = None
        if all(
            not isinstance(linear, WeightOnlyQuantizedLinear) for linear in linear_list
        ):
            weights_list = []
            bias_list = []
            for i in range(self.num_concat):
                weights_list.append(linear_list[i].weight)
                if linear_list[i].bias is not None:
                    bias_list.append(linear_list[i].bias)
            concat_weight = torch.concat(weights_list, 0)
            use_bias = True if bias_list != [] else False
            concat_bias = torch.concat(bias_list, 0) if use_bias else None
            self.concat_linear = nn.Linear(
                concat_weight.shape[1], concat_weight.shape[0], bias=use_bias
            )
            self.concat_linear.weight = nn.Parameter(concat_weight)
            self.concat_linear.bias = nn.Parameter(concat_bias) if use_bias else None

    def forward(self, x):
        output_list = []
        for i in range(self.num_concat):
            assert hasattr(self, f"linear_{i}")
            linear = getattr(self, f"linear_{i}")
            y = linear(x)
            output_list.append(y)
        return tuple(output_list)

    def extra_repr(self):
        return f"num_concat = {self.num_concat}"


class _IPEXlinearSiluMulRef(nn.Module):
    def __init__(self, module_s, module_m):
        super().__init__()
        self.linear_s = module_s
        self.linear_m = module_m

    def forward(self, x):
        return nn.functional.silu(self.linear_s(x)) * self.linear_m(x)
