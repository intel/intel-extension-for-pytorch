import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa F401

import numpy as np

np.set_printoptions(threshold=np.inf)

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_index_put(self, dtype=torch.bfloat16):
        cpu_device = torch.device("cpu")
        dpcpp_device = torch.device("xpu")

        accumulate = True
        x_cpu = torch.zeros([4, 512, 4], dtype=dtype, device=cpu_device)
        y_cpu = torch.ones([4, 15000, 4], dtype=dtype, device=cpu_device)
        x_dpcpp = x_cpu.to(dpcpp_device)
        y_dpcpp = y_cpu.to(dpcpp_device)
        index_cpu = [
            torch.ones([4, 15000, 4], device=cpu_device).to(torch.long),
            torch.ones([4, 15000, 4], device=cpu_device).to(torch.long),
            torch.ones([4, 15000, 4], device=cpu_device).to(torch.long),
        ]
        index_dpcpp = [
            torch.ones([4, 15000, 4], device=dpcpp_device).to(torch.long),
            torch.ones([4, 15000, 4], device=dpcpp_device).to(torch.long),
            torch.ones([4, 15000, 4], device=dpcpp_device).to(torch.long),
        ]

        z_cpu = x_cpu.index_put_(index_cpu, y_cpu, accumulate)

        z_xpu = x_dpcpp.index_put_(index_dpcpp, y_dpcpp, accumulate)

        print("z_cpu = ", z_cpu)
        print("z_xpu = ", z_xpu.cpu())
        self.assertEqual(z_cpu, z_xpu.cpu())
