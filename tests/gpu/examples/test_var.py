import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_var(self, dtype=torch.float):
        src = torch.randn((3, 4,), dtype=dtype, device=cpu_device)
        print("cpu src = ", src)
        print("cpu dst = ", src.var())
        print("cpu dst with dim = ", src.var(1))

        src_dpcpp = src.to("xpu")
        print("gpu src = ", src_dpcpp.cpu())
        print("gpu dst = ", src_dpcpp.var().cpu())
        print("gpu dst with dim = ", src_dpcpp.var(1).cpu())
        self.assertEqual(src, src_dpcpp.to(cpu_device))
        self.assertEqual(src.var(), src_dpcpp.var().to(cpu_device))
        self.assertEqual(src.var(1), src_dpcpp.var(1).to(cpu_device))

        self.assertEqual(src, src_dpcpp.to(cpu_device))
        self.assertEqual(torch.var(src), torch.var(src_dpcpp).to(cpu_device))
        self.assertEqual(torch.var(src, 1), torch.var(src_dpcpp, 1).to(cpu_device))