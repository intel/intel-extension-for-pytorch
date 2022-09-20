import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

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

    def test_var_correction(self, dtype=torch.float):
        src = torch.randn((3, 4,), dtype=dtype, device=cpu_device)
        cpu = torch.empty(4, device=cpu_device)
        torch.var(src, 0, unbiased=True, out=cpu)
        cpu_corr = torch.empty(4, device=cpu_device)
        torch.var(src, 0, correction=3, out=cpu_corr)

        src_dpcpp = src.to("xpu")
        xpu = torch.empty(4, device=dpcpp_device)
        torch.var(src_dpcpp, 0, unbiased=True, out=xpu)
        self.assertEqual(cpu, xpu.to("cpu"))
        xpu_corr = torch.empty(4, device=dpcpp_device)
        torch.var(src_dpcpp, 0, correction=3, out=xpu_corr)
        self.assertEqual(cpu_corr, xpu_corr.to("cpu"))
