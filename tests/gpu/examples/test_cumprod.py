import torch
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_cumprod(self, dtype=torch.float):
        x1 = torch.randn(10, device=cpu_device)
        x1_dpcpp = x1.to(dpcpp_device)
        print("(10) cpu", torch.cumprod(x1, dim=0))
        print("(10) dpcpp", torch.cumprod(x1_dpcpp, dim=0).cpu())
        self.assertEqual(torch.cumprod(x1, dim=0),
                         torch.cumprod(x1_dpcpp, dim=0).to(cpu_device))

        x1_dpcpp = x1.to(dpcpp_device).to(torch.float16)
        # print("(10) half cpu", torch.cumprod(x1, dim=0))
        print("(10) half dpcpp", torch.cumprod(x1_dpcpp, dim=0).cpu())

        x2 = torch.randn(3, 2, 4, device=cpu_device)
        x2_dpcpp = x2.to(dpcpp_device)
        print("(3, 2, 4) cpu", torch.cumprod(x2, dim=0))
        print("(3, 2, 4) dpcpp", torch.cumprod(x2_dpcpp, dim=0).cpu())
        self.assertEqual(torch.cumprod(x2, dim=0),
                         torch.cumprod(x2_dpcpp, dim=0).to(cpu_device))

        x3 = torch.randn(4, 2, 4, device=cpu_device)
        x3_dpcpp = x3.to(cpu_device)
        print("(4, 2, 4) cpu", torch.cumprod(x3, dim=2))
        print("(4, 2, 4) dpcpp", torch.cumprod(x3_dpcpp, dim=2).cpu())
        self.assertEqual(torch.cumprod(x3, dim=2),
                         torch.cumprod(x3_dpcpp, dim=2).to(cpu_device))
