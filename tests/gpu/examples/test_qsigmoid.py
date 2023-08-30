import torch
import intel_extension_for_pytorch  # noqa


from torch.testing._internal.common_utils import TestCase
import platform

class TestTorchMethod(TestCase):
    def test_qsigmoid(self, dtype=torch.float):
        zp_vec = [0] if platform.system() == 'Windows' else [0, 2]
        for dtype in [torch.quint8, torch.qint8]:
            for zp in zp_vec:
                dtype = torch.qint8

                input0 = torch.randn(1, 1, 5, 5, device="xpu")
                q_input = torch.quantize_per_tensor(input0, 0.4, zp, dtype=dtype)
                q_input_cpu = torch.quantize_per_tensor(input0.to("cpu"), 0.4, zp, dtype=dtype)

                qy_cpu = torch.sigmoid(q_input_cpu)
                qy_xpu = torch.sigmoid(q_input)

                self.assertEqual(torch.dequantize(qy_cpu), torch.dequantize(qy_xpu))
