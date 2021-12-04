import numpy
import torch
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):

    def test_reduce_ops(self, dtype=torch.float):
        user_cpu = torch.randn([256, 3, 2, 4], device=cpu_device)
        user_xpu = user_cpu.to("xpu")
        res_cpu = torch._aminmax(user_cpu)
        print("begin dpcpp compute:")
        res_dpcpp = torch._aminmax(user_xpu)
        print(res_cpu[0])
        print(res_cpu[1])
        print(res_dpcpp[0].cpu())
        print(res_dpcpp[1].cpu())
        self.assertEqual(res_cpu[0], res_dpcpp[0].cpu())
        self.assertEqual(res_cpu[1], res_dpcpp[1].cpu())

        # test cache
        print("begin testing primitive cache")
        for i in range(5):
            user_cpu = torch.randn([256, 3, 2, 4], device=cpu_device)
            user_xpu = user_cpu.to("xpu")
            res_cpu = torch._aminmax(user_cpu)
            res_dpcpp = torch._aminmax(user_xpu)
            self.assertEqual(res_cpu[0], res_dpcpp[0].cpu())
            self.assertEqual(res_cpu[1], res_dpcpp[1].cpu())
