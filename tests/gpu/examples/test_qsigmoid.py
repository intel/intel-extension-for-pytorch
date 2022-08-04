import torch
import intel_extension_for_pytorch # noqa
import pytest

from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    @pytest.mark.skip(reason="Need create a PR for torch to align output_scale")
    def test_qsigmoid(self, dtype=torch.float):
        dtype = torch.qint8

        input0 = torch.randn(1, 1, 5, 5, device="xpu")
        q_input = torch.quantize_per_tensor(input0, 0.4, 0, dtype=dtype)
        q_input_cpu = torch.quantize_per_tensor(input0.to("cpu"), 0.4, 0, dtype=dtype)
        result_functional = torch.dequantize(torch.sigmoid(q_input))
        result_inplace = torch.dequantize(torch.sigmoid_(q_input))
        result_output = torch.randn(1, 1, 5, 5, device="xpu")
        result_out = torch.dequantize(torch.sigmoid(q_input, out=result_output))
        result_cpu = torch.dequantize(torch.sigmoid(q_input_cpu))

        self.assertEqual(result_functional.to("cpu"), result_cpu)
        self.assertEqual(result_inplace.to("cpu"), result_cpu)
        self.assertEqual(result_out.to("cpu"), result_cpu)
