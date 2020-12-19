import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_mean(self, dtype=torch.float):
        user_cpu = torch.randn([2, 2, 2, 2, 2], device=cpu_device)
        print(user_cpu)
        res_cpu = torch.mean(user_cpu, (0, 4), False)
        print("cpu result:")
        print(res_cpu)
        print("begin dpcpp compute:")
        res_dpcpp = torch.mean(user_cpu.to("xpu"), (0, 4), False)
        print("dpcpp result:")
        print(res_dpcpp.cpu())
        self.assertEqual(res_cpu, res_dpcpp.cpu())
