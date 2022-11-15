import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_adaptive_max_pool2d(self, dtype=torch.float):
        x_cpu = torch.randn([1, 1, 8, 8], device=cpu_device)
        grad_cpu = torch.randn([1, 1, 2, 2], device=cpu_device)
        x_dpcpp = x_cpu.to(dpcpp_device)
        grad_dpcpp = grad_cpu.to(dpcpp_device)

        self.assertEqual(x_cpu, x_dpcpp.to(cpu_device))
        self.assertEqual(grad_cpu, grad_dpcpp.to(cpu_device))

        max_pool = nn.AdaptiveMaxPool2d((2, 2), return_indices=True)
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        output_cpu = y_cpu[0].backward(grad_cpu)
        print("x_cpu.grad", x_cpu.grad)

        x_dpcpp.requires_grad_(True)
        max_pool = max_pool.to(dpcpp_device)
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].cpu())
        output_dpcpp = y_dpcpp[0].backward(grad_dpcpp)
        print("x_dpcpp.grad", x_dpcpp.grad.cpu())

        self.assertEqual(y_cpu[0], y_dpcpp[0].to(dpcpp_device))
        # For now our MaxPooling return indices are wrong, oneDNN is trying to fix them
        # JIRA: https://jira.devtools.intel.com/browse/MFDNN-3672
        # self.assertEqual( y_cpu[1], y_dpcpp[1].to(dpcpp_device))
        self.assertEqual(x_dpcpp.grad, x_dpcpp.grad.to(cpu_device))

    def test_channels_last_simple_fwd(self, dtype=torch.float):
        x_cpu = torch.randn([3, 2, 8, 8], device=cpu_device)
        x_dpcpp = x_cpu.to(dpcpp_device).to(memory_format=torch.channels_last)
        max_pool = nn.AdaptiveMaxPool2d((2, 2), return_indices=True)

        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        max_pool = max_pool.to(dpcpp_device)
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].cpu())
        self.assertEqual(y_cpu[0], y_dpcpp[0].to(dpcpp_device))

    def test_channels_last_simple_bwd(self, dtype=torch.float):
        x_cpu = torch.randn([3, 2, 8, 8], device=cpu_device)
        grad_cpu = torch.randn([3, 2, 2, 2], device=cpu_device)
        x_dpcpp = x_cpu.to(dpcpp_device).to(memory_format=torch.channels_last)
        grad_dpcpp = grad_cpu.to(dpcpp_device)

        max_pool = nn.AdaptiveMaxPool2d((2, 2), return_indices=True)
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        output_cpu = y_cpu[0].backward(grad_cpu)
        print("x_cpu.grad", x_cpu.grad)

        x_dpcpp.requires_grad_(True)
        max_pool = max_pool.to(dpcpp_device)
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].cpu())
        output_dpcpp = y_dpcpp[0].backward(grad_dpcpp)
        print("x_dpcpp.grad", x_dpcpp.grad.cpu())

        self.assertEqual(y_cpu[0], y_dpcpp[0].to(dpcpp_device))
        # For now our MaxPooling return the local indices
        # While pytorch need the global indices
        # This issue will not be fixed on oneDNN side
        # https://jira.devtools.intel.com/browse/MFDNN-3672
        # We need to consider the DPCPP pooling
        # self.assertEqual(y_cpu[1], y_dpcpp[1].to(dpcpp_device))
        self.assertEqual(x_dpcpp.grad, x_dpcpp.grad.to(cpu_device))

    def test_adaptive_max_pool_3D(self, dtype=torch.float):
        x = torch.randn([30, 40, 50])
        grad = torch.randn([30, 2, 2])
        mem_format = torch.channels_last
        m = nn.AdaptiveMaxPool2d((2, 2), return_indices=True)

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
        grad = torch.randn([40, 2, 2])
        input_cpu = x.clone().transpose(0, 1)
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone()
        output_cpu = m(input_cpu)
        output_cpu[0].backward(grad_cpu)

        # XPU
        input_xpu = x.clone().transpose(0, 1).to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu[0].backward(grad_xpu)

        self.assertEqual(output_cpu[0], output_xpu[0].to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.to(cpu_device))

    def test_adaptive_max_pool_4D(self, dtype=torch.float):
        x = torch.randn([20, 30, 40, 50])
        grad = torch.randn([20, 30, 2, 2])
        m = nn.AdaptiveMaxPool2d((2, 2), return_indices=True)

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
        input_cpu = x.clone()
        input_cpu.requires_grad_(True)
        grad_cpu = grad.clone()
        output_cpu = m(input_cpu)
        output_cpu[0].backward(grad_cpu)

        # XPU
        input_xpu = x.clone().contiguous(memory_format=mem_format).to(dpcpp_device)
        input_xpu.requires_grad_(True)
        grad_xpu = grad.clone().contiguous(memory_format=mem_format).to(dpcpp_device)
        output_xpu = m(input_xpu)
        output_xpu[0].backward(grad_xpu)

        self.assertEqual(output_cpu[0], output_xpu[0].contiguous().to(cpu_device))
        self.assertEqual(input_cpu.grad, input_xpu.grad.contiguous().to(cpu_device))

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
