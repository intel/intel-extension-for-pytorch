import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_conv_tbc(self, dtype=torch.float):

        input_cpu = torch.randn(3, 4, 5)
        weight_cpu = torch.randn(3, 5, 4)
        bias_cpu = torch.randn(4)

        input_xpu = input_cpu.to("xpu")
        weight_xpu = weight_cpu.to("xpu")
        bias_xpu = bias_cpu.to("xpu")

        m = torch.conv_tbc

        input_cpu.requires_grad = True
        output_cpu = m(input_cpu, weight_cpu, bias_cpu)
        output_cpu.backward(torch.ones_like(output_cpu))
        # input_cpu.grad.zero_()

        input_xpu.requires_grad = True
        output_xpu = m(input_xpu, weight_xpu, bias_xpu)
        output_xpu.backward(torch.ones_like(output_xpu).to("xpu"))
        # input_xpu.grad.zero_()

        self.assertEqual(output_cpu, output_xpu.cpu())
        self.assertEqual(input_cpu.grad, input_xpu.grad.cpu())
