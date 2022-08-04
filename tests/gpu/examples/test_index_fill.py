import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_index_fill(self, dtype=torch.float):
        x = torch.ones([5, 3], device=cpu_device)
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, -1, 2])
        x.index_fill_(0, index, -2)
        print("x = ", x)

        x_dpcpp = torch.ones([5, 3], device=dpcpp_device)
        index = torch.tensor([0, 4, 2], device=dpcpp_device)
        x_dpcpp.index_fill_(0, index, -2)

        print("x_dpcpp = ", x_dpcpp.to("cpu"))
        self.assertEqual(x, x_dpcpp.to(cpu_device))


    def test_mult_dim(self, dtype=torch.float):
        a_cpu = torch.randn(11, 22, 1025, 22, 11)
        a_xpu = a_cpu.to(dpcpp_device)
        idx_cpu = torch.tensor([0, 31, 33, 63, 65, -31, -1], dtype=torch.long)
        idx_xpu = idx_cpu.to(dpcpp_device)

        b_cpu = a_cpu.index_fill(2, idx_cpu, 1.11111)
        b_xpu = a_xpu.index_fill(2, idx_xpu, 1.11111)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))


    def test_mult_transposed_dim(self, dtype=torch.float):
        a_cpu = torch.randn(11, 22, 1024, 22, 11)
        a_xpu = a_cpu.to(dpcpp_device)
        idx_cpu = torch.tensor([0, 31, 33, 63, 65, -31, -1], dtype=torch.long)
        idx_xpu = idx_cpu.to(dpcpp_device)

        a_cpu = a_cpu.transpose(1, 3)
        a_xpu = a_xpu.transpose(1, 3)
        b_cpu = a_cpu.index_fill(2, idx_cpu, 1.11111)
        b_xpu = a_xpu.index_fill(2, idx_xpu, 1.11111)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))


    def test_mult_strided_dim(self, dtype=torch.float):
        a_cpu = torch.randn(8, 8, 8)
        a_xpu = a_cpu.to(dpcpp_device)
        idx_cpu = torch.tensor([0, -2, -1], dtype=torch.long)
        idx_xpu = idx_cpu.to(dpcpp_device)

        a_strided_cpu = torch.as_strided(a_cpu, (4, 4, 4), (128, 16, 2))
        a_strided_xpu = torch.as_strided(a_xpu, (4, 4, 4), (128, 16, 2))

        b_cpu = a_strided_cpu.index_fill(2, idx_cpu, 1.11111)
        b_xpu = a_strided_xpu.index_fill(2, idx_xpu, 1.11111)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))

        b_cpu = a_strided_cpu.index_fill(1, idx_cpu, 1.11111)
        b_xpu = a_strided_xpu.index_fill(1, idx_xpu, 1.11111)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))

        b_cpu = a_strided_cpu.index_fill(0, idx_cpu, 1.11111)
        b_xpu = a_strided_xpu.index_fill(0, idx_xpu, 1.11111)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device))
