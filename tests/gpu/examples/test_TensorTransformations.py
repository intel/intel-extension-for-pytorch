import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_flip(self, dtype=torch.float):
        x_cpu = torch.arange(8192, dtype=dtype, device=cpu_device).view(8, 32, 32)
        x_dpcpp = x_cpu.to(dpcpp_device)
        y_cpu = x_cpu.flip([1, 2])
        y_dpcpp = x_dpcpp.flip([1, 2])
        self.assertEqual(y_cpu, y_dpcpp.cpu())

        x_cpu = torch.arange(8192, dtype=dtype, device=cpu_device).view(32, 8, 32)
        x_dpcpp = x_cpu.to(dpcpp_device)
        y_cpu = x_cpu.flip([0, 2])
        y_dpcpp = x_dpcpp.flip([0, 2])
        self.assertEqual(y_cpu, y_dpcpp.cpu())

        x_cpu = torch.arange(8192, dtype=dtype, device=cpu_device).view(32, 32, 8)
        x_dpcpp = x_cpu.to(dpcpp_device)
        y_cpu = x_cpu.flip([0, 1])
        y_dpcpp = x_dpcpp.flip([0, 1])
        self.assertEqual(y_cpu, y_dpcpp.cpu())

    def test_roll(self, dtype=torch.float):
        x_cpu = torch.arange(1024*128*128, dtype=dtype, device=cpu_device).view(1024, 128, 128)
        x_dpcpp = x_cpu.to(dpcpp_device)
        # test 1
        y_cpu = x_cpu.roll(1, 0)
        y_dpcpp = x_dpcpp.roll(1, 0)
        self.assertEqual(y_cpu, y_dpcpp.cpu())
        # test 2
        y_cpu = x_cpu.roll(-1, 0)
        y_dpcpp = x_dpcpp.roll(-1, 0)
        self.assertEqual(y_cpu, y_dpcpp.cpu())
        # test 3
        y_cpu = x_cpu.roll(shifts=(2, 1), dims=(0, 1))
        y_dpcpp = x_dpcpp.roll(shifts=(2, 1), dims=(0, 1))
        self.assertEqual(y_cpu, y_dpcpp.cpu())

    def test_rot90(self, dtype=torch.float):
        x_cpu = torch.arange(8, dtype=dtype, device=cpu_device).view(2, 2, 2)
        x_dpcpp = x_cpu.to(dpcpp_device)
        y_cpu = x_cpu.rot90(1, [1, 2])
        y_dpcpp = x_dpcpp.rot90(1, [1, 2])
        print("test_rot90:")
        print("cpu:")
        print(y_cpu)
        print("dpcpp:")
        print(y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())
