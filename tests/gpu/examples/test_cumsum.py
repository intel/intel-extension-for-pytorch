import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_cumsum_1d_dim0(self, dtype=torch.float):
        print("\n-----------------------1D, dim=0---------------------------------------------")
        x1 = torch.randn(65, device=cpu_device)
        x1_dpcpp = x1.to(dpcpp_device).to(dtype)
        self.assertEqual(torch.cumsum(x1, dim=0), torch.cumsum(
            x1_dpcpp, dim=0).to(cpu_device).to(torch.float), rtol=1.3e-6, atol=1e-5)

    def test_cumsum_1d_dim0_half(self, dtype=torch.half):
        print("\n-----------------------1D, dim=0 half---------------------------------------------")
        x1 = torch.randn(65, device=cpu_device)
        x1_dpcpp = x1.to(dpcpp_device).to(dtype)
        self.assertEqual(torch.cumsum(x1, dim=0), torch.cumsum(
            x1_dpcpp, dim=0).to(cpu_device).to(torch.float), rtol=10e-4, atol=10e-2)

    def test_cumsum_2d_dim0(self, dtype=torch.float):
        print("\n-----------------------2D, dim=0---------------------------------------------")
        x2 = torch.randn(65, 65, device=cpu_device)
        x2_dpcpp = x2.to(dpcpp_device).to(dtype)
        self.assertEqual(torch.cumsum(x2, dim=0), torch.cumsum(
            x2_dpcpp, dim=0).to(cpu_device).to(torch.float), rtol=1.3e-6, atol=1e-5)

    def test_cumsum_2d_dim0_half(self, dtype=torch.half):
        print("\n-----------------------2D, dim=0 half---------------------------------------------")
        x2 = torch.randn(65, 65, device=cpu_device)
        x2_dpcpp = x2.to(dpcpp_device).to(dtype)
        self.assertEqual(torch.cumsum(x2, dim=0), torch.cumsum(
            x2_dpcpp, dim=0).to(cpu_device).to(torch.float), rtol=10e-4, atol=10e-2)

    def test_cumsum_2d_dim1(self, dtype=torch.float):
        print("\n-----------------------2D, dim=1---------------------------------------------")
        x2 = torch.randn(65, 65, device=cpu_device)
        x2_dpcpp = x2.to(dpcpp_device).to(dtype)
        self.assertEqual(torch.cumsum(x2, dim=1), torch.cumsum(
            x2_dpcpp, dim=1).to(cpu_device).to(torch.float), rtol=1.3e-6, atol=1e-5)

    def test_cumsum_2d_dim1_half(self, dtype=torch.half):
        print("\n-----------------------2D, dim=1 half---------------------------------------------")
        x2 = torch.randn(65, 65, device=cpu_device)
        x2_dpcpp = x2.to(dpcpp_device).to(dtype)
        self.assertEqual(torch.cumsum(x2, dim=1), torch.cumsum(
            x2_dpcpp, dim=1).to(cpu_device).to(torch.float), rtol=10e-4, atol=10e-2)

    def test_cumsum_3d_dim0(self, dtype=torch.float):
        print("\n-----------------------3D, dim=0---------------------------------------------")
        x2 = torch.randn(33, 65, 65, device=cpu_device)
        x2_dpcpp = x2.to(dpcpp_device).to(dtype)
        self.assertEqual(torch.cumsum(x2, dim=0), torch.cumsum(
            x2_dpcpp, dim=0).to(cpu_device).to(torch.float), rtol=1.3e-6, atol=1e-5)

    def test_cumsum_3d_dim0_half(self, dtype=torch.half):
        print("\n-----------------------3D, dim=0 half---------------------------------------------")
        x2 = torch.randn(33, 65, 65, device=cpu_device)
        x2_dpcpp = x2.to(dpcpp_device).to(dtype)
        self.assertEqual(torch.cumsum(x2, dim=0), torch.cumsum(
            x2_dpcpp, dim=0).to(cpu_device).to(torch.float), rtol=10e-4, atol=10e-2)

    def test_cumsum_3d_dim1(self, dtype=torch.float):
        print("\n-----------------------3D, dim=1---------------------------------------------")
        x2 = torch.randn(65, 33, 65, device=cpu_device)
        x2_dpcpp = x2.to(dpcpp_device).to(dtype)
        self.assertEqual(torch.cumsum(x2, dim=1), torch.cumsum(
            x2_dpcpp, dim=1).to(cpu_device).to(torch.float), rtol=1.3e-6, atol=1e-5)

    def test_cumsum_3d_dim1_half(self, dtype=torch.half):
        print("\n-----------------------3D, dim=1 half---------------------------------------------")
        x2 = torch.randn(65, 33, 65, device=cpu_device)
        x2_dpcpp = x2.to(dpcpp_device).to(dtype)
        self.assertEqual(torch.cumsum(x2, dim=1), torch.cumsum(
            x2_dpcpp, dim=1).to(cpu_device).to(torch.float), rtol=10e-4, atol=10e-2)

    def test_cumsum_3d_dim2(self, dtype=torch.float):
        print("\n-----------------------3D, dim=2---------------------------------------------")
        x2 = torch.randn(65, 65, 33, device=cpu_device)
        x2_dpcpp = x2.to(dpcpp_device).to(dtype)
        self.assertEqual(torch.cumsum(x2, dim=2), torch.cumsum(
            x2_dpcpp, dim=2).to(cpu_device).to(torch.float), rtol=1.3e-6, atol=1e-5)

    def test_cumsum_3d_dim2_half(self, dtype=torch.half):
        print("\n-----------------------3D, dim=2---------------------------------------------")
        x2 = torch.randn(65, 65, 33, device=cpu_device)
        x2_dpcpp = x2.to(dpcpp_device).to(dtype)
        self.assertEqual(torch.cumsum(x2, dim=2), torch.cumsum(
            x2_dpcpp, dim=2).to(cpu_device).to(torch.float), rtol=10e-4, atol=10e-2)
