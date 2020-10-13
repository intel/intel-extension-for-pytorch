import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    def test_cat_array(self, dtype=torch.float):
        user_cpu1 = torch.randn([2, 2, 3], device=cpu_device)
        user_cpu2 = torch.randn([2, 2, 3], device=cpu_device)
        user_cpu3 = torch.randn([2, 2, 3], device=cpu_device)

        res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=1)
        print("CPU Result:")
        print(res_cpu)

        res_dpcpp = torch.cat((user_cpu1.to(dpcpp_device), user_cpu2.to(
            dpcpp_device), user_cpu3.to(dpcpp_device)), dim=1)
        print("SYCL Result:")
        print(res_dpcpp.cpu())
        self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))
