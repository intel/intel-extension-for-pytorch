import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
import platform


class TestTorchMethod(TestCase):
    def test_q_add(self, dtype=torch.float):
        # Refer to PyTorch's UT in pytorch/test/quantization/core/test_quantized_op.py
        for dtype in [torch.quint8, torch.qint8]:
            add_relu = torch.ops.quantized.add_relu
            add = torch.ops.quantized.add

            A = torch.arange(-128, 130, dtype=torch.float)
            B = torch.arange(-128, 130, dtype=torch.float)
            scale = 2.0
            zero_point = 0 if platform.system() == 'Windows' else 127
            qA = torch.quantize_per_tensor(A, scale=scale, zero_point=zero_point,
                                           dtype=dtype)
            qB = torch.quantize_per_tensor(B, scale=scale, zero_point=zero_point,
                                           dtype=dtype)

            A_xpu = A.to("xpu")
            B_xpu = B.to("xpu")
            qA_xpu = torch.quantize_per_tensor(A, scale=scale, zero_point=zero_point,
                                               dtype=dtype)
            qB_xpu = torch.quantize_per_tensor(B, scale=scale, zero_point=zero_point,
                                               dtype=dtype)

            qC = add(qA, qB, scale, zero_point)
            qC_relu = add_relu(qA, qB, scale, zero_point)

            qC_xpu = add(qA_xpu, qB_xpu, scale, zero_point)
            qC_relu_xpu = add_relu(qA_xpu, qB_xpu, scale, zero_point)

            self.assertEqual(torch.dequantize(qC), torch.dequantize(qC_xpu))
            self.assertEqual(torch.dequantize(qC_relu), torch.dequantize(qC_relu_xpu))
