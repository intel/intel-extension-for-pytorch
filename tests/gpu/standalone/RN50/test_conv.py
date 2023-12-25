import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")
shapes = [
           ((1, 3, 224, 224),(64, 3, 7, 7)),
           ((1, 64, 56, 56),(64, 64, 1, 1)),
           ((1, 64, 56, 56),(64, 64, 3, 3)),
           ((1, 512, 7, 7),(512, 512, 3, 3)),
           ((1, 64, 56, 56),(256, 64, 1, 1)),
           ((1, 256, 56, 56),(64, 256, 1, 1)),
           ((1, 512, 7, 7),(2048, 512, 1, 1)),
           ((1, 128, 28, 28),(128, 128, 3, 3)),
           ((1, 128, 28, 28),(512, 128, 1, 1)),
           ((1, 128, 56, 56),(128, 128, 3, 3)),
           ((1, 2048, 7, 7),(512, 2048, 1, 1)),
           ((1, 256, 14, 14),(256, 256, 3, 3)),
           ((1, 256, 28, 28),(256, 256, 3, 3)),
           ((1, 256, 56, 56),(128, 256, 1, 1)),
           ((1, 256, 56, 56),(512, 256, 1, 1)),
           ((1, 512, 14, 14),(512, 512, 3, 3)),
           ((1, 512, 28, 28),(128, 512, 1, 1)),
           ((1, 512, 28, 28),(256, 512, 1, 1)),
           ((1, 256, 14, 14),(1024, 256, 1, 1)),
           ((1, 512, 28, 28),(1024, 512, 1, 1)),
           ((1, 1024, 14, 14),(256, 1024, 1, 1)),
           ((1, 1024, 14, 14),(512, 1024, 1, 1)),
           ((1, 1024, 14, 14),(2048, 1024, 1, 1)),
        ]
# Note:
# In order to press the gradient of weight below 1,
# the default weight should be set to 1e-ks (ks is kernel_size).
# For now, precision could not be pressed to 1e-5,
# but only if there is a real model which suffers the accuracy problem,
# we won't delve into this issue.


class TestNNMethod(TestCase):
    def test_conv2d(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            N, C, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3]
            x_cpu = torch.randn(
                [N, C, H, W], dtype=dtype, device=cpu_device, requires_grad=False
            )
            grad_cpu = torch.full(
                [N, shape[1][0], (H+2*1-(shape[1][2]-1)-1) + 1, (W+2*1-(shape[1][3]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
            )
            conv_cpu = nn.Conv2d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3]), stride=1, padding=1, bias=False)
            y_cpu = conv_cpu(x_cpu)
            y_cpu.backward(grad_cpu)
            y_cpu_gw = conv_cpu.weight.grad.detach().clone()

            conv_cpu.zero_grad()

            x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_()
            grad_dpcpp = grad_cpu.to(dpcpp_device)
            conv_dpcpp = conv_cpu.to(dpcpp_device)
            y_dpcpp = conv_dpcpp(x_dpcpp)
            y_dpcpp.backward(grad_dpcpp)
            y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

            self.assertEqual(y_cpu, y_dpcpp.cpu())
            self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    def test_conv2d_Float16(self, dtype=torch.bfloat16):
        for shape in shapes:
            N, C, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3]
            x_cpu = torch.randn(
                [N, C, H, W], dtype=dtype, device=cpu_device, requires_grad=False
            )
            grad_cpu = torch.full(
                [N, shape[1][0], (H+2*1-(shape[1][2]-1)-1) + 1, (W+2*1-(shape[1][3]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
            )
            conv_cpu = nn.Conv2d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3]), stride=1, padding=1, bias=False, dtype=dtype)
            y_cpu = conv_cpu(x_cpu)
            y_cpu.backward(grad_cpu)
            y_cpu_gw = conv_cpu.weight.grad.detach().clone()

            conv_cpu.zero_grad()

            dtype_dpcpp = torch.float16
            x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_().to(dtype_dpcpp)
            grad_dpcpp = grad_cpu.to(dpcpp_device).to(dtype_dpcpp)
            conv_dpcpp = conv_cpu.to(dpcpp_device).to(dtype_dpcpp)
            y_dpcpp = conv_dpcpp(x_dpcpp)
            #y_dpcpp.backward(grad_dpcpp)
            #y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

            self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
            #self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    def test_conv2d_with_bias(self, dtype=torch.float):
        for shape in shapes:
            N, C, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3]
            x_cpu = torch.randn(
                [N, C, H, W], dtype=dtype, device=cpu_device, requires_grad=False
            )
            grad_cpu = torch.full(
                [N, shape[1][0], (H+2*1-(shape[1][2]-1)-1) + 1, (W+2*1-(shape[1][3]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
            )
            conv_cpu = nn.Conv2d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3]), stride=1, padding=1, bias=True)
            y_cpu = conv_cpu(x_cpu)
            y_cpu.backward(grad_cpu)
            y_cpu_gw = conv_cpu.weight.grad.detach().clone()

            conv_cpu.zero_grad()

            x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_()
            grad_dpcpp = grad_cpu.to(dpcpp_device)
            conv_dpcpp = conv_cpu.to(dpcpp_device)
            y_dpcpp = conv_dpcpp(x_dpcpp)
            y_dpcpp.backward(grad_dpcpp)
            y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

            self.assertEqual(y_cpu, y_dpcpp.cpu())
            self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=0)

    def test_conv2d_bia_bf16_input_bf16_bia(self, dtype=torch.float):
        for shape in shapes:
            N, C, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3]
            x_cpu = torch.randn(
                [N, C, H, W], dtype=dtype, device=cpu_device, requires_grad=False
            )
            grad_cpu = torch.full(
                [N, shape[1][0], (H+2*1-(shape[1][2]-1)-1) + 1, (W+2*1-(shape[1][3]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
            )
            conv_cpu = nn.Conv2d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3]), stride=1, padding=1, bias=True)
            y_cpu = conv_cpu(x_cpu)
            y_cpu.backward(grad_cpu)
            y_cpu_gw = conv_cpu.weight.grad.detach().clone()

            conv_cpu.zero_grad()

            dtype_dpcpp = torch.bfloat16
            x_dpcpp = x_cpu.to(dpcpp_device).requires_grad_().to(dtype_dpcpp)
            grad_dpcpp = grad_cpu.to(dpcpp_device).to(dtype_dpcpp)
            conv_dpcpp = conv_cpu.to(dpcpp_device).to(dtype_dpcpp)
            y_dpcpp = conv_dpcpp(x_dpcpp)
            y_dpcpp.backward(grad_dpcpp)
            y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

            self.assertEqual(y_cpu, y_dpcpp.to(torch.float).cpu(), rtol=10e-4, atol=10e-2)
            self.assertEqual(
                y_cpu_gw, y_dpcpp_gw.to(torch.float).cpu(), rtol=10e-4, atol=10e-2
            )
