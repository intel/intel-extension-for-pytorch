import torch
import torch.nn as nn
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_adaptive_max_pool3d(self, dtype=torch.float):
        x_cpu = torch.randn([1, 1, 4, 4, 4], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to(dpcpp_device)
        grad_cpu = torch.randn([1, 1, 2, 2, 2], device=cpu_device)
        grad_dpcpp = grad_cpu.to(dpcpp_device)

        self.assertEqual(x_cpu, x_dpcpp.to(cpu_device))
        self.assertEqual(grad_cpu, grad_dpcpp.to(cpu_device))

        max_pool = nn.AdaptiveMaxPool3d((2, 2, 2), return_indices=True)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        y_cpu[0].backward(grad_cpu)
        print("y_cpu backward", x_cpu.grad)

        max_pool.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = max_pool(x_dpcpp)

        print("y_dpcpp", y_dpcpp[0].cpu())
        grad_dpcpp = grad_cpu.to("xpu")
        y_dpcpp[0].backward(grad_dpcpp)
        print("y_dpcpp backward", x_dpcpp.grad.cpu())

        # self.assertEqual( y_cpu, y_dpcpp.to(cpu_device))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_channels_last_simple_fwd(self, dtype=torch.float):
        x_cpu = torch.randn([1, 4, 4, 4, 6], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to(dpcpp_device).to(memory_format=torch.channels_last_3d)
        max_pool = nn.AdaptiveMaxPool3d((2, 2, 2), return_indices=True)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])

        max_pool.to("xpu")
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].cpu())
        self.assertEqual(y_cpu[0], y_dpcpp[0].to(cpu_device))

    def test_channels_last_simple_bwd(self, dtype=torch.float):
        x_cpu = torch.randn([1, 4, 4, 4, 6], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to(dpcpp_device).to(memory_format=torch.channels_last_3d)
        grad_cpu = torch.randn([1, 4, 2, 2, 2], device=cpu_device)
        grad_dpcpp = grad_cpu.to(dpcpp_device)

        max_pool = nn.AdaptiveMaxPool3d((2, 2, 2), return_indices=True)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        y_cpu[0].backward(grad_cpu)
        print("y_cpu backward", x_cpu.grad)

        max_pool.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = max_pool(x_dpcpp)

        print("y_dpcpp", y_dpcpp[0].cpu())
        grad_dpcpp = grad_cpu.to("xpu")
        y_dpcpp[0].backward(grad_dpcpp)
        print("y_dpcpp backward", x_dpcpp.grad.cpu())

        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    @repeat_test_for_types([torch.float, torch.bfloat16])
    def test_adaptive_max_pool3d_4D(self, dtype=torch.float):
        x = torch.randn([20, 30, 40, 50])
        grad = torch.randn([20, 2, 2, 2])
        mem_format = torch.channels_last
        m = nn.AdaptiveMaxPool3d((2, 2, 2), return_indices=True)

        # 4D contiguous input
        # CPU
        input_cpu = x.clone()
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone()
        output_cpu = m(input_cpu)
        output_cpu[0].backward(grad_cpu)

        # XPU
        input_xpu = x.clone().to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu[0].backward(grad_xpu)

        self.assertEqual(output_cpu[0], output_xpu[0].to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))

        # 4D channel_last input
        # CPU
        input_cpu = x.clone().contiguous(memory_format=mem_format)
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone().contiguous(memory_format=mem_format)
        output_cpu = m(input_cpu)
        output_cpu[0].backward(grad_cpu)

        # XPU
        input_xpu = x.clone().contiguous(memory_format=mem_format).to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().contiguous(memory_format=mem_format).to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu[0].backward(grad_xpu)

        self.assertEqual(output_cpu[0], output_xpu[0].to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))

        # 4D non-contiguous input
        # CPU
        input_cpu = x.clone().transpose(2, 3)
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone().transpose(1, 2)
        output_cpu = m(input_cpu)
        output_cpu[0].backward(grad_cpu)

        # XPU
        input_xpu = x.clone().transpose(2, 3).to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().transpose(1, 2).to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu[0].backward(grad_xpu)

        self.assertEqual(output_cpu[0], output_xpu[0].to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))

    @repeat_test_for_types([torch.float, torch.bfloat16])
    def test_adaptive_max_pool3d_5D(self, dtype=torch.float):
        x = torch.randn([10, 20, 30, 40, 50])
        grad = torch.randn([10, 20, 2, 2, 2])
        mem_format = torch.channels_last_3d
        m = nn.AdaptiveMaxPool3d((2, 2, 2), return_indices=True)

        # 5D contiguous input
        # CPU
        input_cpu = x.clone()
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone()
        output_cpu = m(input_cpu)
        output_cpu[0].backward(grad_cpu)

        # XPU
        input_xpu = x.clone().to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu[0].backward(grad_xpu)

        self.assertEqual(output_cpu[0], output_xpu[0].to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))

        # 5D channel_last input
        # CPU
        input_cpu = x.clone().contiguous(memory_format=mem_format)
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone().contiguous(memory_format=mem_format)
        output_cpu = m(input_cpu)
        output_cpu[0].backward(grad_cpu)

        # XPU
        input_xpu = x.clone().contiguous(memory_format=mem_format).to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().contiguous(memory_format=mem_format).to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu[0].backward(grad_xpu)

        self.assertEqual(output_cpu[0], output_xpu[0].to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))

        # 5D non-contiguous input
        # CPU
        input_cpu = x.clone().transpose(2, 3)
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone().transpose(3, 4)
        output_cpu = m(input_cpu)
        output_cpu[0].backward(grad_cpu)

        # XPU
        input_xpu = x.clone().transpose(2, 3).to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().transpose(3, 4).to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu[0].backward(grad_xpu)

        self.assertEqual(output_cpu[0], output_xpu[0].to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))
