import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import platform


class TestNNMethod(TestCase):
    def test_QLeakReLU(self, dtype=torch.float):
        zp_vec = [0] if platform.system() == 'Windows' else [0, 2]
        for dtype in [torch.quint8, torch.qint8]:
            for zp in zp_vec:
                scale = 0.04
                x_cpu = torch.randn([1, 1, 3, 4], device=torch.device("cpu"))
                x_gpu = x_cpu.to("xpu")
                Xelu = nn.LeakyReLU(0.1, inplace=True)

                q_cpu = torch.quantize_per_tensor(x_cpu, scale, zp, dtype)
                y_cpu = Xelu(q_cpu)

                Xelu.to("xpu")
                q_gpu = torch.quantize_per_tensor(x_gpu, scale, zp, dtype)
                y_gpu = Xelu(q_gpu)

                self.assertEqual(y_cpu, y_gpu)
