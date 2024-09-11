import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestNNMethod(TestCase):
    def test_conv_1(self, dtype=torch.float):
        shape = ((1, 128, 112, 112, 80), (64, 128, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-4, rtol=1e-5)

    def test_conv_bfloat16_1(self, dtype=torch.bfloat16):
        shape = ((1, 128, 112, 112, 80), (64, 128, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5e-3, rtol=5e-3)

    def test_conv_float16_1(self, dtype=torch.bfloat16):
        shape = ((1, 128, 112, 112, 80), (64, 128, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_2(self, dtype=torch.float):
        shape = ((1, 128, 56, 56, 40), (128, 128, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_2(self, dtype=torch.bfloat16):
        shape = ((1, 128, 56, 56, 40), (128, 128, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_2(self, dtype=torch.bfloat16):
        shape = ((1, 128, 56, 56, 40), (128, 128, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_3(self, dtype=torch.float):
        shape = ((1, 128, 56, 56, 40), (256, 128, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_3(self, dtype=torch.bfloat16):
        shape = ((1, 128, 56, 56, 40), (256, 128, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_3(self, dtype=torch.bfloat16):
        shape = ((1, 128, 56, 56, 40), (256, 128, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_4(self, dtype=torch.float):
        shape = ((1, 128, 56, 56, 40), (4, 128, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_4(self, dtype=torch.bfloat16):
        shape = ((1, 128, 56, 56, 40), (4, 128, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_4(self, dtype=torch.bfloat16):
        shape = ((1, 128, 56, 56, 40), (4, 128, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_5(self, dtype=torch.float):
        shape = ((1, 256, 28, 28, 20), (256, 256, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_5(self, dtype=torch.bfloat16):
        shape = ((1, 256, 28, 28, 20), (256, 256, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_5(self, dtype=torch.bfloat16):
        shape = ((1, 256, 28, 28, 20), (256, 256, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_6(self, dtype=torch.float):
        shape = ((1, 256, 28, 28, 20), (320, 256, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_6(self, dtype=torch.bfloat16):
        shape = ((1, 256, 28, 28, 20), (320, 256, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_6(self, dtype=torch.bfloat16):
        shape = ((1, 256, 28, 28, 20), (320, 256, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_7(self, dtype=torch.float):
        shape = ((1, 256, 28, 28, 20), (4, 256, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_7(self, dtype=torch.bfloat16):
        shape = ((1, 256, 28, 28, 20), (4, 256, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_7(self, dtype=torch.bfloat16):
        shape = ((1, 256, 28, 28, 20), (4, 256, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_8(self, dtype=torch.float):
        shape = ((1, 256, 56, 56, 40), (128, 256, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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

        self.assertEqual(y_cpu, y_dpcpp.cpu(), atol=5 * 1e-5, rtol=1e-5)
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_8(self, dtype=torch.bfloat16):
        shape = ((1, 256, 56, 56, 40), (128, 256, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_8(self, dtype=torch.bfloat16):
        shape = ((1, 256, 56, 56, 40), (128, 256, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_9(self, dtype=torch.float):
        shape = ((1, 320, 14, 14, 10), (320, 320, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_9(self, dtype=torch.bfloat16):
        shape = ((1, 320, 14, 14, 10), (320, 320, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_9(self, dtype=torch.bfloat16):
        shape = ((1, 320, 14, 14, 10), (320, 320, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_10(self, dtype=torch.float):
        shape = ((1, 320, 14, 14, 10), (4, 320, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_10(self, dtype=torch.bfloat16):
        shape = ((1, 320, 14, 14, 10), (4, 320, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_10(self, dtype=torch.bfloat16):
        shape = ((1, 320, 14, 14, 10), (4, 320, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_11(self, dtype=torch.float):
        shape = ((1, 320, 7, 7, 5), (320, 320, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_11(self, dtype=torch.bfloat16):
        shape = ((1, 320, 7, 7, 5), (320, 320, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_11(self, dtype=torch.bfloat16):
        shape = ((1, 320, 7, 7, 5), (320, 320, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_12(self, dtype=torch.float):
        shape = ((1, 32, 224, 224, 160), (32, 32, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-4, rtol=1e-4)

    def test_conv_bfloat16_12(self, dtype=torch.bfloat16):
        shape = ((1, 32, 224, 224, 160), (32, 32, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_12(self, dtype=torch.bfloat16):
        shape = ((1, 32, 224, 224, 160), (32, 32, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=5*1e-2, rtol=5e-2)

    def test_conv_13(self, dtype=torch.float):
        shape = ((1, 32, 224, 224, 160), (4, 32, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_13(self, dtype=torch.bfloat16):
        shape = ((1, 32, 224, 224, 160), (4, 32, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_13(self, dtype=torch.bfloat16):
        shape = ((1, 32, 224, 224, 160), (4, 32, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_14(self, dtype=torch.float):
        shape = ((1, 32, 224, 224, 160), (64, 32, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-4, rtol=1e-5)

    def test_conv_bfloat16_14(self, dtype=torch.bfloat16):
        shape = ((1, 32, 224, 224, 160), (64, 32, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5*1e-3, rtol=5*1e-3)

    def test_conv_float16_14(self, dtype=torch.bfloat16):
        shape = ((1, 32, 224, 224, 160), (64, 32, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-2)

    def test_conv_15(self, dtype=torch.float):
        shape = ((1, 4, 224, 224, 160), (32, 4, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_15(self, dtype=torch.bfloat16):
        shape = ((1, 4, 224, 224, 160), (32, 4, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_15(self, dtype=torch.bfloat16):
        shape = ((1, 4, 224, 224, 160), (32, 4, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=5*1e-2, rtol=5e-2)

    def test_conv_16(self, dtype=torch.float):
        shape = ((1, 512, 28, 28, 20), (256, 512, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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

        self.assertEqual(y_cpu, y_dpcpp.cpu(), atol=5 * 1e-5, rtol=1e-5)
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_16(self, dtype=torch.bfloat16):
        shape = ((1, 512, 28, 28, 20), (256, 512, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_16(self, dtype=torch.bfloat16):
        shape = ((1, 512, 28, 28, 20), (256, 512, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_17(self, dtype=torch.float):
        shape = ((1, 640, 14, 14, 10), (320, 640, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_17(self, dtype=torch.bfloat16):
        shape = ((1, 640, 14, 14, 10), (320, 640, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_17(self, dtype=torch.bfloat16):
        shape = ((1, 640, 14, 14, 10), (320, 640, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_18(self, dtype=torch.float):
        shape = ((1, 64, 112, 112, 80), (128, 64, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_18(self, dtype=torch.bfloat16):
        shape = ((1, 64, 112, 112, 80), (128, 64, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_18(self, dtype=torch.bfloat16):
        shape = ((1, 64, 112, 112, 80), (128, 64, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_19(self, dtype=torch.float):
        shape = ((1, 64, 112, 112, 80), (4, 64, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_19(self, dtype=torch.bfloat16):
        shape = ((1, 64, 112, 112, 80), (4, 64, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-3)

    def test_conv_float16_19(self, dtype=torch.bfloat16):
        shape = ((1, 64, 112, 112, 80), (4, 64, 1, 1, 1))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)

    def test_conv_20(self, dtype=torch.float):
        shape = ((1, 64, 112, 112, 80), (64, 64, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=1e-5)

    def test_conv_bfloat16_20(self, dtype=torch.bfloat16):
        shape = ((1, 64, 112, 112, 80), (64, 64, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-2, rtol=1e-2)

    def test_conv_float16_20(self, dtype=torch.bfloat16):
        shape = ((1, 64, 112, 112, 80), (64, 64, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-3)
    
    def test_conv_21(self, dtype=torch.float):
        shape = ((1, 64, 224, 224, 160), (32, 64, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=5 * 1e-5, rtol=5*1e-2)

    def test_conv_bfloat16_21(self, dtype=torch.bfloat16):
        shape = ((1, 64, 224, 224, 160), (32, 64, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
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
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu(), atol=1e-3, rtol=1e-2)

    def test_conv_float16_21(self, dtype=torch.bfloat16):
        shape = ((1, 64, 224, 224, 160), (32, 64, 3, 3, 3))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        x_cpu = torch.randn(
            [N, C, D, H, W], dtype=dtype, device=cpu_device, requires_grad=False
        )
        grad_cpu = torch.full(
            [N, shape[1][0], (D+2*1-(shape[1][2]-1)-1) + 1, (H+2*1-(shape[1][3]-1)-1) + 1, (W+2*1-(shape[1][4]-1)-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        conv_cpu = nn.Conv3d(shape[1][1], shape[1][0], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=1, padding=1, bias=False, dtype=dtype)
        y_cpu = conv_cpu(x_cpu)
        y_cpu.backward(grad_cpu)
        y_cpu_gw = conv_cpu.weight.grad.detach().clone()

        conv_cpu.zero_grad()
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.to(dtype_dpcpp).to(dpcpp_device).requires_grad_()
        grad_dpcpp = grad_cpu.to(dtype_dpcpp).to(dpcpp_device)
        conv_dpcpp = conv_cpu.to(dtype_dpcpp).to(dpcpp_device)
        y_dpcpp = conv_dpcpp(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        y_dpcpp_gw = conv_dpcpp.weight.grad.detach().clone()

        self.assertEqual(y_cpu, y_dpcpp.cpu().to(torch.bfloat16))
        self.assertEqual(y_cpu_gw, y_dpcpp_gw.cpu().to(torch.bfloat16), atol=2*1e-2, rtol=1e-2)

