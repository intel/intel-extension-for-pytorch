import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):

    def test_reduce_ops(self, dtype=torch.float):
        user_cpu = torch.randn([1, 64, 32], device=cpu_device)
        res_cpu = torch.mean(user_cpu, 2, False)
        res_dpcpp = torch.mean(user_cpu.to("xpu"), 2, False)
        print("dpcpp mean:")
        print(res_dpcpp.cpu())
        self.assertEqual(res_cpu, res_dpcpp.cpu(), atol=5e-3, rtol=1.3e-06)

        user_cpu = torch.randn([2, 64, 32], device=cpu_device)
        res_cpu = torch.mean(user_cpu, 1, False)
        res_dpcpp = torch.mean(user_cpu.to("xpu"), 1, False)
        print("dpcpp mean:")
        print(res_dpcpp.cpu())
        self.assertEqual(res_cpu, res_dpcpp.cpu(), atol=5e-3, rtol=1.3e-06)

        user_cpu = torch.randn([2, 4, 8, 32], device=cpu_device)
        res_cpu = torch.mean(user_cpu, 2, False)
        res_dpcpp = torch.mean(user_cpu.to("xpu"), 2, False)
        print("dpcpp mean:")
        print(res_dpcpp.cpu())
        self.assertEqual(res_cpu, res_dpcpp.cpu(), atol=5e-3, rtol=1.3e-06)

    def test_reduce_shape(self, dtype=torch.float):
        user_cpu = torch.randn([4, 8, 16, 32], device=cpu_device)
        res_cpu = torch.sum(user_cpu, 2, True)
        print("begin dpcpp compute:")
        res_dpcpp = torch.sum(user_cpu.to("xpu"), 2, True)
        print("dpcpp result:")
        print(res_dpcpp.cpu())
        self.assertEqual(res_cpu, res_dpcpp.cpu())

    def test_reduce_noncontiguous(self, dtype=torch.float):
        user_cpu = torch.randn([4, 8, 16, 32], device=cpu_device)
        res_cpu = torch.sum(user_cpu, 2, True)
        print("begin dpcpp compute:")
        res_dpcpp = torch.sum(user_cpu.to(memory_format=torch.channels_last).to("xpu"), 2, True)
        print("dpcpp result:")
        print(res_dpcpp.cpu())
        self.assertEqual(res_cpu, res_dpcpp.cpu())
