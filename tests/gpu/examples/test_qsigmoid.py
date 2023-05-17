import torch
import intel_extension_for_pytorch  # noqa


from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    def test_qsigmoid(self, dtype=torch.float):
        dtype = torch.qint8

        input0 = torch.randn(1, 1, 5, 5, device="xpu")
        q_input = torch.quantize_per_tensor(input0, 0.4, 0, dtype=dtype)
        q_input_cpu = torch.quantize_per_tensor(input0.to("cpu"), 0.4, 0, dtype=dtype)

        result_functional = torch.dequantize(torch.sigmoid(q_input))
        result_inplace = torch.dequantize(torch.sigmoid_(q_input))
        result_output = torch.randn(1, 1, 5, 5, device="xpu")
        result_out = torch.dequantize(torch.sigmoid(q_input, out=result_output))

        dqX = torch.dequantize(q_input_cpu)
        Y_ref = torch.sigmoid(dqX)
        # Here, we quantize output use opaque u8 tensor setting.
        qY_ref = torch.quantize_per_tensor(Y_ref, 1.0 / 255.0 * 2, 0, torch.qint8)
        dqY_ref = qY_ref.dequantize()

        self.assertEqual(result_functional.to("cpu"), dqY_ref)
        self.assertEqual(result_inplace.to("cpu"), dqY_ref)
        self.assertEqual(result_out.to("cpu"), dqY_ref)
