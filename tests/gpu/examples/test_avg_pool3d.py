import torch
import torch.nn as nn
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_avg_pool3d(self, dtype=torch.float):
        x_cpu = torch.ones([1, 8, 8, 24, 24], device=cpu_device)
        grad_cpu = torch.ones([1, 8, 8, 24, 24], device=cpu_device)

        avg_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = avg_pool(x_cpu)
        # print("y_cpu", y_cpu)
        y_cpu.backward(torch.ones([1, 8, 8, 24, 24], device=cpu_device))
        # print("y_cpu backward", x_cpu.grad)

        x_dpcpp = torch.ones([1, 8, 8, 24, 24], device=dpcpp_device,)
        x_dpcpp.requires_grad_(True)
        y_dpcpp = avg_pool(x_dpcpp)

        # print("y_dpcpp", y_dpcpp.cpu())

        # grad_dpcpp = grad_cpu.to("xpu")
        y_dpcpp.backward(torch.ones([1, 8, 8, 24, 24], device=dpcpp_device))
        # print("y_dpcpp backward", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_channels_last_simple_fwd(self, dtype=torch.float):
        x_cpu = torch.ones([8, 8, 24, 24, 24], device=cpu_device)
        x_dpcpp = torch.ones([8, 8, 24, 24, 24], device=dpcpp_device).to(memory_format=torch.channels_last_3d)
        avg_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)

        # cpu
        y_cpu = avg_pool(x_cpu)
        # print("y_cpu", y_cpu)

        y_dpcpp = avg_pool(x_dpcpp)
        # print("y_dpcpp", y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))

    def test_channels_last_simple_bwd(self, dtype=torch.float):
        x_cpu = torch.ones([8, 8, 8, 8, 8], device=cpu_device)
        grad_cpu = torch.ones([8, 8, 8, 8, 8], device=cpu_device)
        x_dpcpp = torch.ones([8, 8, 8, 8, 8], device=dpcpp_device).to(memory_format=torch.channels_last_3d)
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        avg_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = avg_pool(x_cpu)
        # print("y_cpu", y_cpu)
        y_cpu.backward(grad_cpu)
        # print("y_cpu backward", x_cpu.grad)

        x_dpcpp.requires_grad_(True)
        y_dpcpp = avg_pool(x_dpcpp)

        # print("y_dpcpp", y_dpcpp.cpu())
        y_dpcpp.backward(grad_dpcpp)
        # print("y_dpcpp backward", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    @repeat_test_for_types([torch.float, torch.bfloat16])
    def test_adaptive_avg_pool3d_4D(self, dtype=torch.float):
        x = torch.randn([20, 30, 40, 50])
        grad = torch.randn([20, 30, 40, 50])
        mem_format = torch.channels_last
        m = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)

        # 4D contiguous input
        # CPU
        input_cpu = x.clone()
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone()
        output_cpu = m(input_cpu)
        output_cpu.backward(grad_cpu)

        # XPU
        input_xpu = x.clone().to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu.backward(grad_xpu)

        self.assertEqual(output_cpu, output_xpu.to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))

        # 4D channel_last input
        # CPU
        input_cpu = x.clone().contiguous(memory_format=mem_format)
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone().contiguous(memory_format=mem_format)
        output_cpu = m(input_cpu)
        output_cpu.backward(grad_cpu)

        # XPU
        input_xpu = x.clone().contiguous(memory_format=mem_format).to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().contiguous(memory_format=mem_format).to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu.backward(grad_xpu)

        self.assertEqual(output_cpu, output_xpu.to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))

        # 4D non-contiguous input
        # CPU
        input_cpu = x.clone().transpose(2, 3)
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone().transpose(2, 3)
        output_cpu = m(input_cpu)
        output_cpu.backward(grad_cpu)

        # XPU
        input_xpu = x.clone().transpose(2, 3).to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().transpose(2, 3).to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu.backward(grad_xpu)

        self.assertEqual(output_cpu, output_xpu.to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))

    @repeat_test_for_types([torch.float, torch.bfloat16])
    def test_adaptive_avg_pool3d_5D(self, dtype=torch.float):
        x = torch.randn([10, 20, 30, 40, 50])
        grad = torch.randn([10, 20, 30, 40, 50])
        mem_format = torch.channels_last_3d
        m = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)

        # 5D contiguous input
        # CPU
        input_cpu = x.clone()
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone()
        output_cpu = m(input_cpu)
        output_cpu.backward(grad_cpu)

        # XPU
        input_xpu = x.clone().to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu.backward(grad_xpu)

        self.assertEqual(output_cpu, output_xpu.to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))

        # 5D channel_last input
        # CPU
        input_cpu = x.clone().contiguous(memory_format=mem_format)
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone().contiguous(memory_format=mem_format)
        output_cpu = m(input_cpu)
        output_cpu.backward(grad_cpu)

        # XPU
        input_xpu = x.clone().contiguous(memory_format=mem_format).to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().contiguous(memory_format=mem_format).to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu.backward(grad_xpu)

        self.assertEqual(output_cpu, output_xpu.to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))

        # 5D non-contiguous input
        # CPU
        input_cpu = x.clone().transpose(3, 4)
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone().transpose(3, 4)
        output_cpu = m(input_cpu)
        output_cpu.backward(grad_cpu)

        # XPU
        input_xpu = x.clone().transpose(3, 4).to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().transpose(3, 4).to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu.backward(grad_xpu)

        self.assertEqual(output_cpu, output_xpu.to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))
