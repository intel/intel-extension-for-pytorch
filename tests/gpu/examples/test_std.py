import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_std(self, dtype=torch.float):
        src = torch.randn((3, 4,), dtype=torch.float32,
                          device=torch.device("cpu"))
        print("cpu src = ", src)
        print("cpu dst = ", src.std())
        print("cpu dst with dim = ", src.std(1))

        src_dpcpp = src.to("xpu")
        print("gpu src = ", src_dpcpp.cpu())
        print("gpu dst = ", src_dpcpp.std().cpu())
        print("gpu dst with dim = ", src_dpcpp.std(1).cpu())
        self.assertEqual(src, src_dpcpp.to(cpu_device))
        self.assertEqual(src.std(), src_dpcpp.std().to(cpu_device))
        self.assertEqual(src.std(1), src_dpcpp.std(1).to(cpu_device))
        self.assertEqual(torch.std(src), torch.std(src_dpcpp).to(cpu_device))
