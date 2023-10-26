import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_qbinary(self, dtype=torch.float):
        zp_vec = [0, 2]
        for dtype in [torch.quint8, torch.qint8]:
            for zp in zp_vec:
                x_cpu = torch.randn([1, 1, 3, 3], device=cpu_device)
                y_cpu = torch.randn([1, 1, 3, 3], device=cpu_device)

                scale_in_1 = 0.421009
                scale_in_2 = 2.04386

                scale_out = 0.2
                zp_out = 3

                q_x_cpu = torch.quantize_per_tensor(x_cpu, scale_in_1, zp, dtype)
                q_y_cpu = torch.quantize_per_tensor(y_cpu, scale_in_2, zp, dtype)

                x_xpu = x_cpu.to("xpu")
                y_xpu = y_cpu.to("xpu")

                q_x_xpu = torch.quantize_per_tensor(x_xpu, scale_in_1, zp, dtype)
                q_y_xpu = torch.quantize_per_tensor(y_xpu, scale_in_2, zp, dtype)

                ref1 = torch.ops.quantized.add(q_x_cpu, q_y_cpu, scale_out, zp_out)
                real1 = torch.ops.quantized.add(q_x_xpu, q_y_xpu, scale_out, zp_out)

                self.assertEqual(ref1, real1)
