import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


test_1d_shapes = [1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65]
test_2d_shapes = [(1, 7), (7, 1), (2, 7), (2, 8), (2, 9), (7, 1), (7, 2), (8, 2), (9, 2),
                  (3, 15), (3, 16), (3, 17), (15, 3), (16, 3), (17, 3),
                  (4, 31), (4, 32), (4, 33), (31, 4), (32, 4), (33, 4), (65, 65), (8193, 8193)]
test_3d_shapes = [(1, 7, 7), (7, 7, 1), (7, 1, 7), (2, 7, 7), (2, 8, 8), (2, 9, 9),
                  (7, 7, 1), (7, 7, 2), (8, 8, 2), (9, 9, 2), (3, 15, 15), (15, 3, 15),
                  (3, 16, 16), (3, 17, 17), (15, 15, 3), (16, 16, 3), (17, 17, 3),
                  (4, 31, 31), (31, 4, 31), (4, 32, 32), (4, 33, 33), (31, 31, 4),
                  (32, 32, 4), (33, 33, 4), (33, 65, 65), (87, 7983, 45)]
class TestTorchMethod(TestCase):
    def test_cumsum_1d_dim0(self, dtype=torch.float):
        print("\n-----------------------1D, dim=0---------------------------------------------")
        for shape in test_1d_shapes:
            x1 = torch.randn(shape, device=cpu_device)
            x1_dpcpp = x1.to(dpcpp_device).to(dtype)
            if x1.size(0) > 1000:
                atol = 1e-4
            else:
                atol = 1e-5
            self.assertEqual(torch.cumsum(x1, dim=0), torch.cumsum(
                x1_dpcpp, dim=0).to(cpu_device).to(torch.float), rtol=1.3e-6, atol=atol)

    def test_cumsum_1d_dim0_half(self, dtype=torch.half):
        print("\n-----------------------1D, dim=0 half---------------------------------------------")
        for shape in test_1d_shapes:
            x1 = torch.randn(shape, device=cpu_device)
            x1_dpcpp = x1.to(dpcpp_device).to(dtype)
            if x1.size(0) > 100:
                atol = 1
            else:
                atol = 1e-1
            self.assertEqual(torch.cumsum(x1, dim=0), torch.cumsum(
                x1_dpcpp, dim=0).to(cpu_device).to(torch.float), rtol=10e-4, atol=atol)

    def test_cumsum_2d_dim0(self, dtype=torch.float):
        print("\n-----------------------2D, dim=0---------------------------------------------")
        for shape in test_2d_shapes:
            x2 = torch.randn(shape, device=cpu_device)
            x2_dpcpp = x2.to(dpcpp_device).to(dtype)
            if x2.size(0) > 1000:
                atol = 1e-4
            else:
                atol = 1e-5
            self.assertEqual(torch.cumsum(x2, dim=0), torch.cumsum(
                x2_dpcpp, dim=0).to(cpu_device).to(torch.float), rtol=1.3e-6, atol=atol)

    def test_cumsum_2d_dim0_half(self, dtype=torch.half):
        print("\n-----------------------2D, dim=0 half---------------------------------------------")
        for shape in test_2d_shapes:
            x2 = torch.randn(shape, device=cpu_device)
            x2_dpcpp = x2.to(dpcpp_device).to(dtype)
            if x2.size(0) > 100:
                atol = 1
            else:
                atol = 1e-1
            self.assertEqual(torch.cumsum(x2, dim=0), torch.cumsum(
                x2_dpcpp, dim=0).to(cpu_device).to(torch.float), rtol=10e-4, atol=atol)

    def test_cumsum_2d_dim1(self, dtype=torch.float):
        print("\n-----------------------2D, dim=1---------------------------------------------")
        for shape in test_2d_shapes:
            x2 = torch.randn(shape, device=cpu_device)
            x2_dpcpp = x2.to(dpcpp_device).to(dtype)
            if x2.size(1) > 1000:
                atol = 1e-4
            else:
                atol = 1e-5
            self.assertEqual(torch.cumsum(x2, dim=1), torch.cumsum(
                x2_dpcpp, dim=1).to(cpu_device).to(torch.float), rtol=1.3e-6, atol=atol)

    def test_cumsum_2d_dim1_half(self, dtype=torch.half):
        print("\n-----------------------2D, dim=1 half---------------------------------------------")
        for shape in test_2d_shapes:
            x2 = torch.randn(shape, device=cpu_device)
            x2_dpcpp = x2.to(dpcpp_device).to(dtype)
            if x2.size(1) > 100:
                atol = 1
            else:
                atol = 1e-1
            self.assertEqual(torch.cumsum(x2, dim=1), torch.cumsum(
                x2_dpcpp, dim=1).to(cpu_device).to(torch.float), rtol=10e-4, atol=atol)

    def test_cumsum_3d_dim0(self, dtype=torch.float):
        print("\n-----------------------3D, dim=0---------------------------------------------")
        for shape in test_3d_shapes:
            x2 = torch.randn(shape, device=cpu_device)
            x2_dpcpp = x2.to(dpcpp_device).to(dtype)
            if x2.size(0) > 1000:
                atol = 1e-4
            else:
                atol = 1e-5
            self.assertEqual(torch.cumsum(x2, dim=0), torch.cumsum(
                x2_dpcpp, dim=0).to(cpu_device).to(torch.float), rtol=1.3e-6, atol=atol)

    def test_cumsum_3d_dim0_half(self, dtype=torch.half):
        print("\n-----------------------3D, dim=0 half---------------------------------------------")
        for shape in test_3d_shapes:
            x2 = torch.randn(shape, device=cpu_device)
            x2_dpcpp = x2.to(dpcpp_device).to(dtype)
            if x2.size(0) > 100:
                atol = 1
            else:
                atol = 1e-1
            self.assertEqual(torch.cumsum(x2, dim=0), torch.cumsum(
                x2_dpcpp, dim=0).to(cpu_device).to(torch.float), rtol=10e-4, atol=atol)

    def test_cumsum_3d_dim1(self, dtype=torch.float):
        print("\n-----------------------3D, dim=1---------------------------------------------")
        for shape in test_3d_shapes:
            x2 = torch.randn(shape, device=cpu_device)
            x2_dpcpp = x2.to(dpcpp_device).to(dtype)
            if x2.size(1) > 1000:
                atol = 1e-4
            else:
                atol = 1e-5
            self.assertEqual(torch.cumsum(x2, dim=1), torch.cumsum(
                x2_dpcpp, dim=1).to(cpu_device).to(torch.float), rtol=1.3e-6, atol=atol)

    def test_cumsum_3d_dim1_half(self, dtype=torch.half):
        print("\n-----------------------3D, dim=1 half---------------------------------------------")
        for shape in test_3d_shapes:
            x2 = torch.randn(shape, device=cpu_device)
            x2_dpcpp = x2.to(dpcpp_device).to(dtype)
            if x2.size(1) > 100:
                atol = 1
            else:
                atol = 1e-1
            self.assertEqual(torch.cumsum(x2, dim=1), torch.cumsum(
                x2_dpcpp, dim=1).to(cpu_device).to(torch.float), rtol=10e-4, atol=atol)

    def test_cumsum_3d_dim2(self, dtype=torch.float):
        print("\n-----------------------3D, dim=2---------------------------------------------")
        for shape in test_3d_shapes:
            x2 = torch.randn(shape, device=cpu_device)
            x2_dpcpp = x2.to(dpcpp_device).to(dtype)
            if x2.size(2) > 1000:
                atol = 1e-4
            else:
                atol = 1e-5
            self.assertEqual(torch.cumsum(x2, dim=2), torch.cumsum(
                x2_dpcpp, dim=2).to(cpu_device).to(torch.float), rtol=1.3e-6, atol=atol)

    def test_cumsum_3d_dim2_half(self, dtype=torch.half):
        print("\n-----------------------3D, dim=2 half---------------------------------------------")
        for shape in test_3d_shapes:
            x2 = torch.randn(shape, device=cpu_device)
            x2_dpcpp = x2.to(dpcpp_device).to(dtype)
            if x2.size(2) > 100:
                atol = 1
            else:
                atol = 1e-1
            self.assertEqual(torch.cumsum(x2, dim=2), torch.cumsum(
                x2_dpcpp, dim=2).to(cpu_device).to(torch.float), rtol=10e-4, atol=atol)


    def test_mult_transposed_dim(self, dtype=torch.float):
        a_cpu = torch.randn(11, 22, 1024, 22, 11)
        a_xpu = a_cpu.to(dpcpp_device)

        a_cpu = a_cpu.transpose(1, 2)
        a_xpu = a_xpu.transpose(1, 2)
        b_cpu = torch.cumsum(a_cpu, dim=1)
        b_xpu = torch.cumsum(a_xpu, dim=1)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device), rtol=1e-5, atol=1e-4)


    def test_mult_strided_dim(self, dtype=torch.float):
        a_cpu = torch.randn(8, 82, 814)
        a_xpu = a_cpu.to(dpcpp_device)

        a_strided_cpu = torch.as_strided(a_cpu, (4, 41, 407), (2 * 814 * 82, 2 * 814, 2))
        a_strided_xpu = torch.as_strided(a_xpu, (4, 41, 407), (2 * 814 * 82, 2 * 814, 2))

        b_cpu = torch.cumsum(a_strided_cpu, dim=2)
        b_xpu = torch.cumsum(a_strided_xpu, dim=2)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device), rtol=1e-5, atol=1e-4)

        b_cpu = torch.cumsum(a_strided_cpu, dim=1)
        b_xpu = torch.cumsum(a_strided_xpu, dim=1)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device), rtol=1e-5, atol=1e-4)

        b_cpu = torch.cumsum(a_strided_cpu, dim=0)
        b_xpu = torch.cumsum(a_strided_xpu, dim=0)
        self.assertEqual(b_cpu, b_xpu.to(cpu_device), rtol=1e-5, atol=1e-4)
