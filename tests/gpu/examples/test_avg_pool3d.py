import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_avg_pool3d(self, dtype=torch.float):
        x_cpu = torch.ones([1, 8, 24, 24, 24], device=cpu_device)
        grad_cpu = torch.ones([1, 8, 24, 24, 24], device=cpu_device)

        avg_pool = nn.AvgPool3d(kernel_size=12, stride=3, padding=4)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = avg_pool(x_cpu)
        # print("y_cpu", y_cpu)
        y_cpu.backward(torch.ones([1, 8, 7, 7, 7], device=cpu_device))
        # print("y_cpu backward", x_cpu.grad)

        x_dpcpp = torch.ones(
            [1, 8, 24, 24, 24],
            device=dpcpp_device,
        )
        x_dpcpp.requires_grad_(True)
        y_dpcpp = avg_pool(x_dpcpp)

        # print("y_dpcpp", y_dpcpp.cpu())

        # grad_dpcpp = grad_cpu.to("xpu")
        y_dpcpp.backward(torch.ones([1, 8, 7, 7, 7], device=dpcpp_device))
        # print("y_dpcpp backward", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))


    def test_avg_pool3d_large_batch(self, dtype=torch.float):
        x_cpu = torch.ones([10000, 1, 8, 24, 24], device=cpu_device)
        grad_cpu = torch.ones([10000, 1, 8, 24, 24], device=cpu_device)

        avg_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = avg_pool(x_cpu)
        # print("y_cpu", y_cpu)
        y_cpu.backward(torch.ones([10000, 1, 8, 24, 24], device=cpu_device))
        # print("y_cpu backward", x_cpu.grad)

        x_dpcpp = torch.ones(
            [10000, 1, 8, 24, 24],
            device=dpcpp_device,
        )
        x_dpcpp.requires_grad_(True)
        y_dpcpp = avg_pool(x_dpcpp)

        # print("y_dpcpp", y_dpcpp.cpu())

        # grad_dpcpp = grad_cpu.to("xpu")
        y_dpcpp.backward(torch.ones([10000, 1, 8, 24, 24], device=dpcpp_device))
        # print("y_dpcpp backward", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))


    def test_channels_last_simple_fwd(self, dtype=torch.float):
        size_list = [[8, 8, 24, 24, 24], [1, 8, 24, 24, 24]]
        kernel_list = [[3, 1, 1], [12, 3, 4]]
        for size, kernel in zip(size_list, kernel_list):

            x_cpu = torch.ones(size, device=cpu_device)
            x_dpcpp = torch.ones(size, device=dpcpp_device).to(
                memory_format=torch.channels_last_3d
            )
            avg_pool = nn.AvgPool3d(kernel)

            # cpu
            y_cpu = avg_pool(x_cpu)
            # print("y_cpu", y_cpu)

            y_dpcpp = avg_pool(x_dpcpp)
            # print("y_dpcpp", y_dpcpp.cpu())
            self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))


    def test_channels_last_simple_bwd(self, dtype=torch.float):
        x_cpu = torch.ones([8, 8, 8, 8, 8], device=cpu_device)
        grad_cpu = torch.ones([8, 8, 8, 8, 8], device=cpu_device)
        x_dpcpp = torch.ones([8, 8, 8, 8, 8], device=dpcpp_device).to(
            memory_format=torch.channels_last_3d
        )
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

        # 4D strided input - (C,D,H,W) -> (C,H,W,D)
        # CPU
        input_cpu = x.clone().permute(0, 2, 3, 1)
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone().permute(0, 2, 3, 1)
        output_cpu = m(input_cpu)
        output_cpu.backward(grad_cpu)

        # XPU
        input_xpu = x.clone().permute(0, 2, 3, 1).to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().permute(0, 2, 3, 1).to(dpcpp_device)
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

    def test_avg_pool3d_blk_format(self, dtype=torch.float):
        x = torch.randn([10, 16, 30, 40, 50])
        grad = torch.randn([10, 16, 30, 40, 50])
        conv_cpu1 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        pool_cpu = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
        conv_cpu2 = nn.Conv3d(16, 16, kernel_size=1, stride=1, bias=False)

        # 5D contiguous input
        # CPU
        input_cpu = x.clone()
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone()
        output_cpu = conv_cpu2(pool_cpu(conv_cpu1(input_cpu)))
        output_cpu.backward(grad_cpu)

        conv_cpu1.zero_grad()
        conv_cpu2.zero_grad()

        # XPU
        with torch.xpu.onednn_layout():
            input_xpu = x.clone().to(dpcpp_device)
            input_xpu.requires_grad_(True)
            grad_xpu = grad.clone().to(dpcpp_device)
            conv_dpcpp1 = conv_cpu1.to(dpcpp_device)
            pool_dpcpp = pool_cpu.to(dpcpp_device)
            conv_dpcpp2 = conv_cpu2.to(dpcpp_device)
            output_xpu = conv_dpcpp2(pool_dpcpp(conv_dpcpp1(input_xpu)))
            output_xpu.backward(grad_xpu)

        self.assertEqual(output_cpu, output_xpu.to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))
