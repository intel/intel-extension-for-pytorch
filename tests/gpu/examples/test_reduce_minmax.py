import numpy
import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch

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

    def test_min_max_focus_case(self, dtype=torch.float):
        a = torch.randn(4, 15000)
        res_min_cpu = torch.min(a, -1)
        res_max_cpu = torch.max(a, -1)
        a = a.to("xpu")

        res_min = torch.min(a, -1)
        res_max = torch.max(a, -1)
        self.assertEqual(res_min_cpu[0], res_min[0].cpu())
        self.assertEqual(res_min_cpu[1], res_min[1].cpu())
        self.assertEqual(res_max_cpu[0], res_max[0].cpu())
        self.assertEqual(res_max_cpu[1], res_max[1].cpu())

        a = torch.randn(2000000, 2)
        res_min_cpu = torch.min(a, 0)
        res_max_cpu = torch.max(a, 0)
        a = a.to("xpu")

        res_min = torch.min(a, 0)
        res_max = torch.max(a, 0)
        self.assertEqual(res_min_cpu[0], res_min[0].cpu())
        self.assertEqual(res_min_cpu[1], res_min[1].cpu())
        self.assertEqual(res_max_cpu[0], res_max[0].cpu())
        self.assertEqual(res_max_cpu[1], res_max[1].cpu())
