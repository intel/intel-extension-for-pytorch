import torch
from torch.nn.modules.utils import _pair
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import pytest


class TestTorchMethod(TestCase):
    def test_qhardtanh(self, dtype=torch.float):

        zero_point = 0

        dtype_inputs = torch.quint8

        inputs = torch.randn(5, 5)

        q_inputs = torch.quantize_per_tensor(inputs, 0.4, zero_point, dtype_inputs)

        output_int8 = torch.nn.quantized.functional.hardtanh(q_inputs)

        print("start xpu")
        inputs_gpu = inputs.to("xpu")

        q_inputs_gpu = torch.quantize_per_tensor(inputs_gpu, 0.4, zero_point, dtype_inputs)

        output_gpu_int8 = torch.nn.quantized.functional.hardtanh(q_inputs_gpu)


        self.assertEqual(output_int8, output_gpu_int8)

if __name__ == "__main__":
    mod = TestTorchMethod()
    mod.test_qhardtanh()
