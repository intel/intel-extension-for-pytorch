import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_bernoulli(self, dtype=torch.float):
        user_cpu = torch.empty(3, 3, dtype=dtype).uniform_(0, 1)
        dpcpp_res = torch.bernoulli(user_cpu.to(dpcpp_device))
        print("dpcpp Res")
        print(dpcpp_res.cpu())

        # Examine the output contains only 1 and 0
        self.assertTrue(torch.ne(dpcpp_res.to(cpu_device), 0).mul_(
            torch.ne(dpcpp_res.to(cpu_device), 1)).sum().item() == 0)
        # self.assertTrue(True, torch.ne(dpcpp_res.to(cpu_device), 0).mul_(
        #    torch.ne(dpcpp_res.to(cpu_device), 1)).sum().item() == 0)
