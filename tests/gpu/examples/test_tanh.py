import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestNNMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_tanh(self, dtype=torch.float):

        # cpu
        tanh = nn.Tanh()

        x_cpu = torch.tensor([[1.23, 2.34, 6.45, 2.22], [0.23, 1.34, 7.45, 1.22]],
                             requires_grad=True, dtype=dtype)

        z_cpu = tanh(x_cpu)

        z_cpu.backward(torch.tensor([[1, 1, 1, 1], [2, 2, 3, 4]], dtype=dtype))

        # dpcpp
        x_dpcpp = torch.tensor([[1.23, 2.34, 6.45, 2.22], [
            0.23, 1.34, 7.45, 1.22]], requires_grad=True, device="xpu", dtype=dtype)

        z_dpcpp = tanh(x_dpcpp)

        z_dpcpp.backward(torch.tensor(
            [[1, 1, 1, 1], [2, 2, 3, 4]], device="xpu", dtype=dtype))

        self.assertEqual(x_cpu, x_dpcpp.cpu())
        self.assertEqual(z_cpu, z_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
