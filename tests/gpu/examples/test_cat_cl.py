import torch
from torch.testing._internal.common_utils import TestCase

import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_cat_array(self, dtype=torch.float):
        user_cpu1 = torch.randn([8, 7, 3, 2], device=cpu_device)
        user_cpu2 = torch.randn([8, 7, 3, 2], device=cpu_device)
        user_cpu3 = torch.randn([8, 7, 3, 2], device=cpu_device)

        user_cpu1 = user_cpu1.to(memory_format=torch.channels_last)
        user_cpu2 = user_cpu2.to(memory_format=torch.channels_last)
        user_cpu3 = user_cpu3.to(memory_format=torch.channels_last)

        dim_idx = 1
        res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=dim_idx)
        print("\n-------------CPU Result:--------------")
        print(res_cpu.shape)
        print("res_cpu is cl: ", res_cpu.is_contiguous(memory_format=torch.channels_last))
        print(res_cpu)

        user_xpu1 = user_cpu1.to(dpcpp_device)
        user_xpu2 = user_cpu2.to(dpcpp_device)
        user_xpu3 = user_cpu3.to(dpcpp_device)

        print("\n-------------GPU Result:--------------")
        res_dpcpp = torch.cat((user_xpu1, user_xpu2, user_xpu3), dim=dim_idx)
        print("SYCL Result:")
        print(res_dpcpp.cpu().shape)
        print("res_dpcpp is cl: ", res_dpcpp.is_contiguous(memory_format=torch.channels_last))
        print(res_dpcpp.cpu())
        self.assertEqual(res_cpu, res_dpcpp.to(cpu_device))
