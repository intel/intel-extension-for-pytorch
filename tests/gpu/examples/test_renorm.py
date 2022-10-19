import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestNNMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_renorm(self, dtype=torch.float):

        x_cpu = torch.ones(3, 3)

        x_cpu[1].fill_(2)
        x_cpu[2].fill_(3)
        x_dpcpp = x_cpu.to("xpu")

        renorm1 = torch.renorm(x_cpu, 1, 1, 5)

        # print("torch.renorm(x_cpu, 1, 1, 5)", renorm)

        renorm1_dpcpp = torch.renorm(x_dpcpp, 1, 1, 5)
        self.assertEqual(renorm1, renorm1_dpcpp.cpu())

        # print("torch.renorm(x_dpcpp, 1, 1, 5)", renorm.cpu())

        renorm2 = torch.renorm(x_cpu, 1, 0, 5)

        # print("torch.renorm(x_cpu, 1, 0, 5)", renorm)

        renorm2_dpcpp = torch.renorm(x_dpcpp, 1, 0, 5)
        self.assertEqual(renorm2, renorm2_dpcpp.cpu())

        # print("torch.renorm(x_dpcpp, 1, 0, 5)", renorm.cpu())
