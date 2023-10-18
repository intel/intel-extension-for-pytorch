import torch
import intel_extension_for_pytorch as ipex  # noqa F401
from intel_extension_for_pytorch.nn.utils._weight_prepack import (
    _IPEXLinear as _IPEXLinear,
)
import enum


class EltwiseType(enum.IntEnum):
    NotFused = 0
    ReLU = 1
    Sigmoid = 2


class IPEXLinearEltwise(torch.nn.Module):
    def __init__(self, ipex_linear_module, eltwise="relu"):
        super(IPEXLinearEltwise, self).__init__()
        assert isinstance(ipex_linear_module, _IPEXLinear)
        self.m = ipex_linear_module
        self.out_features = ipex_linear_module.out_features
        if eltwise == "relu":
            self.eltwise = EltwiseType.ReLU
        else:
            assert eltwise == "sigmoid"
            self.eltwise = EltwiseType.Sigmoid

    def forward(self, x):
        return torch.ops.torch_ipex.ipex_linear_eltwise(
            x,
            self.m.weight,
            self.m.bias,
            self.eltwise,
            self.m.ctx.get_data_handle(),
            self.out_features,
        )
