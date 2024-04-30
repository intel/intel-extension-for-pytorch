import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

# functionality


class TestNNMethod(TestCase):
    def test_LeakyReLU_1(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float):
        x_cpu = torch.randn(
            [1, 128, 56, 56, 40], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 128, 56, 56, 40, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_bfloat16_1(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.bfloat16):
        x_cpu = torch.randn(
            [1, 128, 56, 56, 40], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 128, 56, 56, 40, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_float16_1(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float16):
        x_cpu = torch.randn(
            [1, 128, 56, 56, 40], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 128, 56, 56, 40, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_2(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float):
        x_cpu = torch.randn(
            [1, 256, 28, 28, 20], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 256, 28, 28, 20, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_bfloat16_2(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.bfloat16):
        x_cpu = torch.randn(
            [1, 256, 28, 28, 20], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 256, 28, 28, 20, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_float16_2(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float16):
        x_cpu = torch.randn(
            [1, 256, 28, 28, 20], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 256, 28, 28, 20, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_3(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float):
        x_cpu = torch.randn(
            [1, 320, 14, 14, 10], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 320, 14, 14, 10, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_bfloat16_3(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.bfloat16):
        x_cpu = torch.randn(
            [1, 320, 14, 14, 10], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 320, 14, 14, 10, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_float16_3(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float16):
        x_cpu = torch.randn(
            [1, 320, 14, 14, 10], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 320, 14, 14, 10, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_4(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float):
        x_cpu = torch.randn(
            [1, 320, 7, 7, 5], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 320, 7, 7, 5, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_bfloat16_4(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.bfloat16):
        x_cpu = torch.randn(
            [1, 320, 7, 7, 5], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 320, 7, 7, 5, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_float16_4(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float16):
        x_cpu = torch.randn(
            [1, 320, 7, 7, 5], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 320, 7, 7, 5, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_5(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float):
        x_cpu = torch.randn(
            [1, 32, 224, 224, 160], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 32, 224, 224, 160, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_bfloat16_5(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.bfloat16):
        x_cpu = torch.randn(
            [1, 32, 224, 224, 160], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 32, 224, 224, 160, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_float16_5(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float16):
        x_cpu = torch.randn(
            [1, 32, 224, 224, 160], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 32, 224, 224, 160, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_6(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float):
        x_cpu = torch.randn(
            [1, 64, 112, 112, 80], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 64, 112, 112, 80, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_bfloat16_6(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.bfloat16):
        x_cpu = torch.randn(
            [1, 64, 112, 112, 80], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 64, 112, 112, 80, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_float16_6(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float16):
        x_cpu = torch.randn(
            [1, 64, 112, 112, 80], device=cpu_device, requires_grad=True, dtype=dtype
        )
        grad_x = torch.randn(
            1, 64, 112, 112, 80, device=cpu_device, requires_grad=True, dtype=dtype
        )
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
