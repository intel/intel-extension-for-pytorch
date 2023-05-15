import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_reduce_ops(self, dtype=torch.float):
        user_cpu = torch.randn([1, 64, 32], device=cpu_device)
        res_cpu = torch.mean(user_cpu, 2, False)
        res_xpu = torch.mean(user_cpu.to("xpu"), 2, False)
        print("xpu mean:")
        print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu(), atol=5e-3, rtol=1.3e-06)

        user_cpu = torch.randn([2, 64, 32], device=cpu_device)
        res_cpu = torch.mean(user_cpu, 1, False)
        res_xpu = torch.mean(user_cpu.to("xpu"), 1, False)
        print("xpu mean:")
        print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu(), atol=5e-3, rtol=1.3e-06)

        user_cpu = torch.randn([2, 4, 8, 32], device=cpu_device)
        res_cpu = torch.mean(user_cpu, 2, False)
        res_xpu = torch.mean(user_cpu.to("xpu"), 2, False)
        print("xpu mean:")
        print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu(), atol=5e-3, rtol=1.3e-06)

    def test_reduce_shape(self, dtype=torch.float):
        user_cpu = torch.randn([4, 8, 16, 32], device=cpu_device)
        res_cpu = torch.sum(user_cpu, 2, True)
        print("begin xpu compute:")
        res_xpu = torch.sum(user_cpu.to("xpu"), 2, True)
        print("xpu result:")
        print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_reduce_noncontiguous(self, dtype=torch.float):
        user_cpu = torch.randn([4, 8, 16, 32], device=cpu_device)
        res_cpu = torch.sum(user_cpu, 2, True)
        print("begin xpu compute:")
        res_xpu = torch.sum(
            user_cpu.to(memory_format=torch.channels_last).to("xpu"), 2, True
        )
        print("xpu result:")
        print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_reduce_global_reduce(self, dtype=torch.float):
        user_cpu = torch.randn([8192, 8192], device=cpu_device)
        res_cpu = user_cpu.sum()
        print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").sum()
        print("xpu result:")
        print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

        user_cpu = torch.randn([8192, 8192], dtype=torch.bfloat16, device=cpu_device)
        res_cpu = user_cpu.sum()
        print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").sum()
        print("xpu result:")
        print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())
