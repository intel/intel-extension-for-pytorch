from pickle import TRUE
import torch
import intel_extension_for_pytorch
import pytest

from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):

    def test_square(self, dtype=torch.float):
        dtype = torch.float

        input0 = torch.randn(8192, 8192, device="xpu")
        input0_cpu = input0.to("cpu")
        result_functional = torch.square(input0)
        result_out = torch.empty(8192, 8192, device="xpu")
        result_out = torch.square(input0, out=result_out)
        result_cpu = torch.square(input0_cpu)

        self.assertEqual(result_functional.to("cpu"), result_cpu)
        self.assertEqual(result_out.to("cpu"), result_cpu)
