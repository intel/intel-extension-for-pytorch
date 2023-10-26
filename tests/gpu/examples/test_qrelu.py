import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa


class TestNNMethod(TestCase):
    def test_QReLU(self, dtype=torch.float):
        zp_vec = [0, 2]
        for dtype in [torch.quint8, torch.qint8]:
            for zp in zp_vec:
                scale = 0.04
                x_cpu = torch.randn([1, 1, 3, 4], device=torch.device("cpu"))
                x_gpu = x_cpu.to("xpu")
                mod = nn.ReLU()

                q_cpu = torch.quantize_per_tensor(x_cpu, scale, zp, dtype)
                y_cpu = mod(q_cpu)

                mod.to("xpu")
                q_gpu = torch.quantize_per_tensor(x_gpu, scale, zp, dtype)
                y_gpu = mod(q_gpu)

                print("y_cpu:", y_cpu)
                print("y_gpu:", y_gpu)

                self.assertEqual(torch.dequantize(y_cpu), torch.dequantize(y_gpu))
