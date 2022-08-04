import torch
import torch.nn as nn
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)

import intel_extension_for_pytorch # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_max_pool3d(self, dtype=torch.float):
        x_cpu = torch.randn([1, 4, 4, 3, 3], device=cpu_device, dtype=dtype)
        grad_cpu = torch.randn([1, 4, 4, 3, 3], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to("xpu")
        grad_dpcpp = grad_cpu.to("xpu")

        max_pool = nn.MaxPool3d(kernel_size=3, stride=1,
                                padding=1, return_indices=True)

        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        output_cpu = y_cpu[0].backward(grad_cpu)
        print("x_cpu.grad", x_cpu.grad)

        max_pool.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].to("cpu"))
        output_dpcpp = y_dpcpp[0].backward(grad_dpcpp)
        print("x_dpcpp.grad", x_dpcpp.grad.to("cpu"))
        self.assertEqual(y_cpu[0], y_dpcpp[0].cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_channels_last_3d_simple_fwd(self, dtype=torch.float):
        x_cpu = torch.randn([1, 4, 4, 3, 3], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to("xpu").to(memory_format=torch.channels_last_3d)

        max_pool = nn.MaxPool3d(kernel_size=3, stride=1,
                                padding=1, return_indices=True)

        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])

        max_pool.to("xpu")
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].to("cpu"))
        self.assertEqual(y_cpu[0], y_dpcpp[0].cpu())

    def test_channels_last_3d_simple_bwd(self, dtype=torch.float):
        x_cpu = torch.randn([1, 4, 4, 3, 3], device=cpu_device, dtype=dtype)
        grad_cpu = torch.randn([1, 4, 4, 3, 3], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to("xpu").to(memory_format=torch.channels_last_3d)
        grad_dpcpp = grad_cpu.to("xpu")

        max_pool = nn.MaxPool3d(kernel_size=3, stride=1,
                                padding=1, return_indices=True)

        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        output_cpu = y_cpu[0].backward(grad_cpu)
        print("x_cpu.grad", x_cpu.grad)

        max_pool.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].to("cpu"))
        output_dpcpp = y_dpcpp[0].backward(grad_dpcpp)
        print("x_dpcpp.grad", x_dpcpp.grad.to("cpu"))
        self.assertEqual(y_cpu[0], y_dpcpp[0].cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    @repeat_test_for_types([torch.float, torch.bfloat16])
    def test_max_pool3d_4D(self, dtype=torch.float):
        x = torch.randn([20, 30, 40, 50])
        grad = torch.randn([20, 30, 40, 50])
        mem_format = torch.channels_last
        m = nn.MaxPool3d(kernel_size=3, stride=1,
                         padding=1, return_indices=True)

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
        grad_cpu = grad.clone().transpose(2, 3)
        output_cpu = m(input_cpu)
        output_cpu[0].backward(grad_cpu)

        # XPU
        input_xpu = x.clone().transpose(2, 3).to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().transpose(2, 3).to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu[0].backward(grad_xpu)

        self.assertEqual(output_cpu[0], output_xpu[0].to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))

    @repeat_test_for_types([torch.float, torch.bfloat16])
    def test_max_pool3d_5D(self, dtype=torch.float):
        x = torch.randn([10, 20, 30, 40, 50])
        grad = torch.randn([10, 20, 30, 40, 50])
        mem_format = torch.channels_last_3d
        m = nn.MaxPool3d(kernel_size=3, stride=1,
                         padding=1, return_indices=True)

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
        input_cpu = x.clone().transpose(3, 4)
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone().transpose(3, 4)
        output_cpu = m(input_cpu)
        output_cpu[0].backward(grad_cpu)

        # XPU
        input_xpu = x.clone().transpose(3, 4).to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().transpose(3, 4).to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu[0].backward(grad_xpu)

        self.assertEqual(output_cpu[0], output_xpu[0].to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))
