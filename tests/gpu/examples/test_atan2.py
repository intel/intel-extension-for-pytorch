import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_atan2(self, dtype=torch.float):
        print("Testing atan2 float:")
        x_cpu = torch.randn(4, device=cpu_device)
        y_cpu = torch.randn(4, device=cpu_device)

        dist_cpu = torch.atan2(x_cpu, y_cpu)

        print("torch.atan2(x_cpu, y_cpu)", dist_cpu)

        x_dpcpp = x_cpu.to(dpcpp_device)
        y_dpcpp = y_cpu.to(dpcpp_device)

        dist_dpcpp = torch.atan2(x_dpcpp, y_dpcpp)

        print("torch.atan2(x_dpcpp, y_dpcpp)", dist_dpcpp.cpu())

        self.assertEqual(x_cpu, x_dpcpp.to(cpu_device))
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
        self.assertEqual(dist_cpu, dist_dpcpp.to(cpu_device))

        print("Testing atan2 int:")
        x_cpu_int = torch.randint(100, (1, 10), dtype=torch.long, device=cpu_device)
        y_cpu_int = torch.randint(100, (1, 10), dtype=torch.long, device=cpu_device)

        dist_cpu_int = torch.atan2(x_cpu_int, y_cpu_int)

        print("torch.atan2(x_cpu_int, y_cpu_int)", dist_cpu_int)

        x_dpcpp_int = x_cpu_int.to(dpcpp_device)
        y_dpcpp_int = y_cpu_int.to(dpcpp_device)

        dist_dpcpp_int = torch.atan2(x_dpcpp_int, y_dpcpp_int)

        print("torch.atan2(x_dpcpp_int, y_dpcpp_int)", dist_dpcpp_int.cpu())

        self.assertEqual(x_cpu_int, x_dpcpp_int.to(cpu_device))
        self.assertEqual(y_cpu_int, y_dpcpp_int.to(cpu_device))
        self.assertEqual(dist_cpu_int, dist_dpcpp_int.to(cpu_device))
