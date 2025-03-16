import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
from intel_extension_for_pytorch.transformers.models.xpu.optimize_transformers.modules.Layers import (
    IPEXLmHeadLinearAllreduceWithPadding,
    IPEXLmHeadLinearAllreduceWithPaddingBaichuan,
)

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_lm_head(self, dtype=torch.float16, device=dpcpp_device):
        input_dim = 4096
        output_dim = 4096
        layer = torch.nn.Linear(
            input_dim, output_dim, bias=True, dtype=dtype, device=dpcpp_device
        )
        input_ori = torch.rand(4096, 1, input_dim, dtype=dtype, device=dpcpp_device)
        input_IPEX = torch.rand(4096, 1, input_dim, dtype=dtype, device=dpcpp_device)
        input_IPEX.copy_(input_ori)
        weight = torch.rand(output_dim, input_dim, dtype=dtype, device=dpcpp_device)
        weight.copy_(layer.weight)

        result = torch.nn.functional.linear(input_ori, weight, layer.bias)

        IPEXLmHead = IPEXLmHeadLinearAllreduceWithPadding(layer)

        result_IPEX = IPEXLmHead(input_IPEX)

        self.assertEqual(result, result_IPEX, atol=0.01, rtol=0.01)

    def test_lm_head_baichuan(self, dtype=torch.float16, device=dpcpp_device):
        input_dim = 4096
        output_dim = 4096
        layer = torch.nn.Linear(
            input_dim, output_dim, bias=True, dtype=dtype, device=dpcpp_device
        )
        input_ori = torch.rand(4096, 1, input_dim, dtype=dtype, device=dpcpp_device)
        input_IPEX = torch.rand(4096, 1, input_dim, dtype=dtype, device=dpcpp_device)
        input_IPEX.copy_(input_ori)
        weight = torch.rand(output_dim, input_dim, dtype=dtype, device=dpcpp_device)
        weight.copy_(layer.weight)

        IPEXLmHead = IPEXLmHeadLinearAllreduceWithPaddingBaichuan(layer)

        weight.data = torch.nn.functional.normalize(weight)
        result = torch.nn.functional.linear(input_ori, weight, layer.bias)

        result_IPEX = IPEXLmHead(input_IPEX)

        self.assertEqual(result, result_IPEX, atol=0.1, rtol=0.1)
