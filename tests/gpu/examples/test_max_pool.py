import torch
import torch.nn as nn
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)

import intel_extension_for_pytorch # noqa
import pytest


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_max_pool(self, dtype=torch.float):
        x_cpu = torch.ones([2, 2, 3, 3], device=cpu_device, dtype=dtype)
        grad_cpu = torch.ones([2, 2, 3, 3], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to("xpu")
        grad_dpcpp = grad_cpu.to("xpu")

        conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=True)
        max_pool = nn.MaxPool2d(kernel_size=3, stride=1,
                                padding=1, return_indices=True)

        x_cpu.requires_grad_(True)
        y_cpu1 = conv1(x_cpu)
        y_cpu = max_pool(y_cpu1)
        print("y_cpu", y_cpu[0])
        output_cpu = y_cpu[0].backward(grad_cpu)
        print("x_cpu.grad", x_cpu.grad)

        conv1.to("xpu")
        max_pool.to("xpu")

        x_dpcpp.requires_grad_(True)
        y_dpcpp1 = conv1(x_dpcpp)
        y_dpcpp = max_pool(y_dpcpp1)
        print("y_dpcpp", y_dpcpp[0].to("cpu"))
        output_dpcpp = y_dpcpp[0].backward(grad_dpcpp)
        print("x_dpcpp.grad", x_dpcpp.grad.to("cpu"))
        self.assertEqual(y_cpu[0], y_dpcpp[0].cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    @pytest.mark.skipif(not torch.xpu.has_channels_last_1d(), reason="doesn't enable channels last 1d")
    def test_channels_last_1d_fwd_and_bwd(self, dtype=torch.float):
        shapes = [(2, 2, 3), (4, 4, 4), (4, 4, 1), (4, 1, 4),
                  (4, 1, 1), (1, 4, 4), (1, 4, 1)]
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, W = shape[0], shape[1], shape[2]
            x_cpu = torch.ones([N, C, W], device=cpu_device, dtype=dtype)
            grad_cpu = torch.ones([N, C, W], device=cpu_device, dtype=dtype)
            x_dpcpp = x_cpu.to("xpu").to(memory_format=torch.channels_last_1d)
            grad_dpcpp = grad_cpu.to("xpu").to(memory_format=torch.channels_last_1d)
            max_pool = nn.MaxPool1d(kernel_size=3, stride=1,
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

            if 1 == y_dpcpp[0].shape[1] or (1 == y_dpcpp[0].shape[2]) or \
               (1 == y_dpcpp[0].shape[1] and 1 == y_dpcpp[0].shape[2]):
                self.assertEqual(y_dpcpp[0].is_contiguous(), True)
                self.assertEqual(y_dpcpp[0].is_contiguous(memory_format=torch.channels_last_1d), True)
            else:
                self.assertEqual(y_dpcpp[0].is_contiguous(), False)
                self.assertEqual(y_dpcpp[0].is_contiguous(memory_format=torch.channels_last_1d), True)

            self.assertEqual(y_cpu[0], y_dpcpp[0].cpu())
            self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_channels_last_fwd_and_bwd(self, dtype=torch.float):
        shapes = [(2, 2, 1, 3), (4, 4, 1, 4), (4, 4, 1, 1), (4, 1, 1, 4),
                  (4, 1, 1, 1), (1, 4, 1, 4), (1, 4, 1, 1)]
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            x_cpu = torch.ones([N, C, H, W], device=cpu_device, dtype=dtype)
            grad_cpu = torch.ones([N, C, H, W], device=cpu_device, dtype=dtype)
            x_dpcpp = x_cpu.to("xpu").to(memory_format=torch.channels_last)
            grad_dpcpp = grad_cpu.to("xpu").to(memory_format=torch.channels_last)

            max_pool = nn.MaxPool2d(kernel_size=3, stride=1,
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

            if 1 == y_dpcpp[0].shape[1] or (1 == y_dpcpp[0].shape[2] and 1 == y_dpcpp[0].shape[3]) or \
               (1 == y_dpcpp[0].shape[1] and 1 == y_dpcpp[0].shape[2] and 1 == y_dpcpp[0].shape[3]):
                self.assertEqual(y_dpcpp[0].is_contiguous(), True)
                self.assertEqual(y_dpcpp[0].is_contiguous(memory_format=torch.channels_last), True)
            else:
                self.assertEqual(y_dpcpp[0].is_contiguous(), False)
                self.assertEqual(y_dpcpp[0].is_contiguous(memory_format=torch.channels_last), True)

            self.assertEqual(y_cpu[0], y_dpcpp[0].cpu())
            self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    @repeat_test_for_types([torch.float, torch.bfloat16])
    def test_max_pool_3D(self, dtype=torch.float):
        x = torch.randn([30, 40, 50])
        grad = torch.randn([30, 40, 50])
        m = nn.MaxPool2d(kernel_size=3, stride=1,
                         padding=1, return_indices=True)

        # 3D contiguous input
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

        # 3D non-contiguous input
        # CPU
        input_cpu = x.clone().transpose(0, 1)
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone().transpose(0, 1)
        output_cpu = m(input_cpu)
        output_cpu[0].backward(grad_cpu)

        # XPU
        input_xpu = x.clone().transpose(0, 1).to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().transpose(0, 1).to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu[0].backward(grad_xpu)

        self.assertEqual(output_cpu[0], output_xpu[0].to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))

    @repeat_test_for_types([torch.float, torch.bfloat16])
    def test_max_pool_4D(self, dtype=torch.float):
        x = torch.randn([20, 30, 40, 50])
        grad = torch.randn([20, 30, 40, 50])
        m = nn.MaxPool2d(kernel_size=3, stride=1,
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
        mem_format = torch.channels_last
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
