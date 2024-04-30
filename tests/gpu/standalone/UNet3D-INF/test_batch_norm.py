import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest
import itertools

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestNNMethod(TestCase):
    def test_batch_norm_1(self, dtype=torch.float):
        shape = (1, 128, 56, 56, 40)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)

        x_dpcpp_i = x_i.to(xpu_device)
        grad_dpcpp_i = grad_i.to(xpu_device)

        self.assertEqual(x_i, x_dpcpp_i.to(cpu_device))
        self.assertEqual(grad_i, grad_dpcpp_i.to(cpu_device))

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn1 = nn.BatchNorm3d(C)
        bn2 = nn.BatchNorm3d(C)
        y_cpu1 = bn1(x_cpu)
        y_cpu = bn2(y_cpu1)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn1.to(xpu_device)
        bn2.to(xpu_device)

        y_dpcpp1 = bn1(x_dpcpp)
        y_dpcpp = bn2(y_dpcpp1)

        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_batch_norm_bfloat16_1(self, dtype=torch.bfloat16):
        shape = (1, 128, 56, 56, 40)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)

        x_dpcpp_i = x_i.to(xpu_device)
        grad_dpcpp_i = grad_i.to(xpu_device)

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn = nn.BatchNorm3d(C)
        y_cpu = bn(x_cpu)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn.to(dtype).to(xpu_device)
        y_dpcpp = bn(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device), rtol=1e-2, atol=1e-2)
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device), rtol=1e-2, atol=1e-2)

    def test_batch_norm_float16_1(self, dtype=torch.float):
        shape = (1, 128, 56, 56, 40)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        x_dpcpp_i = x_i.to(xpu_device).to(dtype).to(dtype_dpcpp)
        grad_dpcpp_i = grad_i.to(xpu_device).to(dtype).to(dtype_dpcpp)

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn = nn.BatchNorm3d(C)
        y_cpu = bn(x_cpu)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn.to(dtype).to(xpu_device)
        y_dpcpp = bn(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device).to(torch.float), rtol=1e-3, atol=1e-3)
        self.assertEqual(
            x_cpu.grad, x_dpcpp.grad.to(cpu_device).to(torch.float), rtol=1e-3, atol=1e-3
        )

    def test_batch_norm_2(self, dtype=torch.float):
        shape = (1, 256, 28, 28, 20)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)

        x_dpcpp_i = x_i.to(xpu_device)
        grad_dpcpp_i = grad_i.to(xpu_device)

        self.assertEqual(x_i, x_dpcpp_i.to(cpu_device))
        self.assertEqual(grad_i, grad_dpcpp_i.to(cpu_device))

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn1 = nn.BatchNorm3d(C)
        bn2 = nn.BatchNorm3d(C)
        y_cpu1 = bn1(x_cpu)
        y_cpu = bn2(y_cpu1)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn1.to(xpu_device)
        bn2.to(xpu_device)

        y_dpcpp1 = bn1(x_dpcpp)
        y_dpcpp = bn2(y_dpcpp1)

        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_batch_norm_bfloat16_2(self, dtype=torch.bfloat16):
        shape = (1, 256, 28, 28, 20)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)

        x_dpcpp_i = x_i.to(xpu_device)
        grad_dpcpp_i = grad_i.to(xpu_device)

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn = nn.BatchNorm3d(C)
        y_cpu = bn(x_cpu)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn.to(dtype).to(xpu_device)
        y_dpcpp = bn(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device), rtol=1e-2, atol=1e-2)
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device), rtol=1e-2, atol=1e-2)

    def test_batch_norm_float16_2(self, dtype=torch.float):
        shape = (1, 256, 28, 28, 20)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        x_dpcpp_i = x_i.to(xpu_device).to(dtype).to(dtype_dpcpp)
        grad_dpcpp_i = grad_i.to(xpu_device).to(dtype).to(dtype_dpcpp)

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn = nn.BatchNorm3d(C)
        y_cpu = bn(x_cpu)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn.to(dtype).to(xpu_device)
        y_dpcpp = bn(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device).to(torch.float), rtol=1e-3, atol=1e-3)
        self.assertEqual(
            x_cpu.grad, x_dpcpp.grad.to(cpu_device).to(torch.float), rtol=1e-3, atol=1e-3
        )

    def test_batch_norm_3(self, dtype=torch.float):
        shape = (1, 320, 14, 14, 10)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)

        x_dpcpp_i = x_i.to(xpu_device)
        grad_dpcpp_i = grad_i.to(xpu_device)

        self.assertEqual(x_i, x_dpcpp_i.to(cpu_device))
        self.assertEqual(grad_i, grad_dpcpp_i.to(cpu_device))

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn1 = nn.BatchNorm3d(C)
        bn2 = nn.BatchNorm3d(C)
        y_cpu1 = bn1(x_cpu)
        y_cpu = bn2(y_cpu1)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn1.to(xpu_device)
        bn2.to(xpu_device)

        y_dpcpp1 = bn1(x_dpcpp)
        y_dpcpp = bn2(y_dpcpp1)

        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_batch_norm_bfloat16_3(self, dtype=torch.bfloat16):
        shape = (1, 320, 14, 14, 10)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)

        x_dpcpp_i = x_i.to(xpu_device)
        grad_dpcpp_i = grad_i.to(xpu_device)

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn = nn.BatchNorm3d(C)
        y_cpu = bn(x_cpu)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn.to(dtype).to(xpu_device)
        y_dpcpp = bn(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device), rtol=1e-2, atol=1e-2)
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device), rtol=1e-2, atol=1e-2)

    def test_batch_norm_float16_3(self, dtype=torch.float):
        shape = (1, 320, 14, 14, 10)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        x_dpcpp_i = x_i.to(xpu_device).to(dtype).to(dtype_dpcpp)
        grad_dpcpp_i = grad_i.to(xpu_device).to(dtype).to(dtype_dpcpp)

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn = nn.BatchNorm3d(C)
        y_cpu = bn(x_cpu)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn.to(dtype).to(xpu_device)
        y_dpcpp = bn(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device).to(torch.float), rtol=1e-3, atol=1e-3)
        self.assertEqual(
            x_cpu.grad, x_dpcpp.grad.to(cpu_device).to(torch.float), rtol=1e-3, atol=1e-3
        )

    def test_batch_norm_4(self, dtype=torch.float):
        shape = (1, 320, 7, 7, 5)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)

        x_dpcpp_i = x_i.to(xpu_device)
        grad_dpcpp_i = grad_i.to(xpu_device)

        self.assertEqual(x_i, x_dpcpp_i.to(cpu_device))
        self.assertEqual(grad_i, grad_dpcpp_i.to(cpu_device))

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn1 = nn.BatchNorm3d(C)
        bn2 = nn.BatchNorm3d(C)
        y_cpu1 = bn1(x_cpu)
        y_cpu = bn2(y_cpu1)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn1.to(xpu_device)
        bn2.to(xpu_device)

        y_dpcpp1 = bn1(x_dpcpp)
        y_dpcpp = bn2(y_dpcpp1)

        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_batch_norm_bfloat16_4(self, dtype=torch.bfloat16):
        shape = (1, 320, 7, 7, 5)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)

        x_dpcpp_i = x_i.to(xpu_device)
        grad_dpcpp_i = grad_i.to(xpu_device)

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn = nn.BatchNorm3d(C)
        y_cpu = bn(x_cpu)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn.to(dtype).to(xpu_device)
        y_dpcpp = bn(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device), rtol=1e-2, atol=1e-2)
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device), rtol=1e-2, atol=1e-2)

    def test_batch_norm_float16_4(self, dtype=torch.float):
        shape = (1, 320, 7, 7, 5)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        x_dpcpp_i = x_i.to(xpu_device).to(dtype).to(dtype_dpcpp)
        grad_dpcpp_i = grad_i.to(xpu_device).to(dtype).to(dtype_dpcpp)

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn = nn.BatchNorm3d(C)
        y_cpu = bn(x_cpu)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn.to(dtype).to(xpu_device)
        y_dpcpp = bn(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device).to(torch.float), rtol=1e-3, atol=1e-3)
        self.assertEqual(
            x_cpu.grad, x_dpcpp.grad.to(cpu_device).to(torch.float), rtol=1e-3, atol=1e-3
        )

    def test_batch_norm_5(self, dtype=torch.float):
        shape = (1, 32, 224, 224, 160)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)

        x_dpcpp_i = x_i.to(xpu_device)
        grad_dpcpp_i = grad_i.to(xpu_device)

        self.assertEqual(x_i, x_dpcpp_i.to(cpu_device))
        self.assertEqual(grad_i, grad_dpcpp_i.to(cpu_device))

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn1 = nn.BatchNorm3d(C)
        bn2 = nn.BatchNorm3d(C)
        y_cpu1 = bn1(x_cpu)
        y_cpu = bn2(y_cpu1)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn1.to(xpu_device)
        bn2.to(xpu_device)

        y_dpcpp1 = bn1(x_dpcpp)
        y_dpcpp = bn2(y_dpcpp1)

        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_batch_norm_bfloat16_5(self, dtype=torch.bfloat16):
        shape = (1, 32, 224, 224, 160)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)

        x_dpcpp_i = x_i.to(xpu_device)
        grad_dpcpp_i = grad_i.to(xpu_device)

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn = nn.BatchNorm3d(C)
        y_cpu = bn(x_cpu)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn.to(dtype).to(xpu_device)
        y_dpcpp = bn(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device), rtol=1e-2, atol=1e-2)
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device), rtol=1e-2, atol=1e-2)

    def test_batch_norm_float16_5(self, dtype=torch.float):
        shape = (1, 32, 224, 224, 160)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        x_dpcpp_i = x_i.to(xpu_device).to(dtype).to(dtype_dpcpp)
        grad_dpcpp_i = grad_i.to(xpu_device).to(dtype).to(dtype_dpcpp)

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn = nn.BatchNorm3d(C)
        y_cpu = bn(x_cpu)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn.to(dtype).to(xpu_device)
        y_dpcpp = bn(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device).to(torch.float), rtol=1e-3, atol=1e-3)
        self.assertEqual(
            x_cpu.grad, x_dpcpp.grad.to(cpu_device).to(torch.float), rtol=1e-3, atol=1e-3
        )

    def test_batch_norm_6(self, dtype=torch.float):
        shape = (1, 64, 112, 112, 80)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)

        x_dpcpp_i = x_i.to(xpu_device)
        grad_dpcpp_i = grad_i.to(xpu_device)

        self.assertEqual(x_i, x_dpcpp_i.to(cpu_device))
        self.assertEqual(grad_i, grad_dpcpp_i.to(cpu_device))

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn1 = nn.BatchNorm3d(C)
        bn2 = nn.BatchNorm3d(C)
        y_cpu1 = bn1(x_cpu)
        y_cpu = bn2(y_cpu1)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn1.to(xpu_device)
        bn2.to(xpu_device)

        y_dpcpp1 = bn1(x_dpcpp)
        y_dpcpp = bn2(y_dpcpp1)

        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_batch_norm_bfloat16_6(self, dtype=torch.bfloat16):
        shape = (1, 64, 112, 112, 80)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)

        x_dpcpp_i = x_i.to(xpu_device)
        grad_dpcpp_i = grad_i.to(xpu_device)

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn = nn.BatchNorm3d(C)
        y_cpu = bn(x_cpu)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn.to(dtype).to(xpu_device)
        y_dpcpp = bn(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device), rtol=1e-2, atol=1e-2)
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device), rtol=1e-2, atol=1e-2)

    def test_batch_norm_float16_6(self, dtype=torch.float):
        shape = (1, 64, 112, 112, 80)
        N, C, D, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        grad_i = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        x_dpcpp_i = x_i.to(xpu_device).to(dtype).to(dtype_dpcpp)
        grad_dpcpp_i = grad_i.to(xpu_device).to(dtype).to(dtype_dpcpp)

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn = nn.BatchNorm3d(C)
        y_cpu = bn(x_cpu)

        y_cpu.backward(grad_cpu)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn.to(dtype).to(xpu_device)
        y_dpcpp = bn(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device).to(torch.float), rtol=1e-3, atol=1e-3)
        self.assertEqual(
            x_cpu.grad, x_dpcpp.grad.to(cpu_device).to(torch.float), rtol=1e-3, atol=1e-3
        )
    
