
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa


class TestTorchMethod(TestCase):
    def test_deconv1d_bias(self, dtype=torch.float):
        deconv = nn.ConvTranspose1d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)

        x_cpu = torch.randn(2, 16, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 128)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to('xpu')

        x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to('xpu')
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv1d_bias_blk(self, dtype=torch.float):
        deconv = nn.ConvTranspose1d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)

        x_cpu = torch.randn(2, 16, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 128)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        with torch.xpu.onednn_layout():
            deconv.zero_grad()
            deconv = deconv.to('xpu')

            x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
            x_xpu.retain_grad()
            gy_xpu = gy_cpu.to('xpu')
            y_xpu = deconv(x_xpu)
            y_xpu.backward(gy_xpu)
            gw_xpu = deconv.weight.grad
            gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_group_deconv1d_bias(self, dtype=torch.float):
        deconv = nn.ConvTranspose1d(16, 32, kernel_size=3, stride=1, padding=1, groups=2, bias=True)

        x_cpu = torch.randn(2, 16, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 128)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to('xpu')

        x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to('xpu')
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_group_deconv1d_bias_blk(self, dtype=torch.float):
        deconv = nn.ConvTranspose1d(16, 32, kernel_size=3, stride=1, padding=1, groups=2, bias=True)

        x_cpu = torch.randn(2, 16, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 128)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        with torch.xpu.onednn_layout():
            deconv.zero_grad()
            deconv = deconv.to('xpu')

            x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
            x_xpu.retain_grad()
            gy_xpu = gy_cpu.to('xpu')
            y_xpu = deconv(x_xpu)
            # FIXME:
            # crash with current oneDNN, Floating point exception (core dumped)
            # and will be fixed after oneDNN upgraded to internal master branch with commit df0b87c2e14
            # y_xpu.backward(gy_xpu)
            # gw_xpu = deconv.weight.grad
            # gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        # self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        # self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        # self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv2d_bias(self, dtype=torch.float):
        deconv = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)

        x_cpu = torch.randn(2, 16, 128, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 128, 128)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to('xpu')

        x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to('xpu')
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv2d_bias_blk(self, dtype=torch.float):
        deconv = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)

        x_cpu = torch.randn(2, 16, 128, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 128, 128)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        with torch.xpu.onednn_layout():
            deconv.zero_grad()
            deconv = deconv.to('xpu')

            x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
            x_xpu.retain_grad()
            gy_xpu = gy_cpu.to('xpu')
            y_xpu = deconv(x_xpu)
            y_xpu.backward(gy_xpu)
            gw_xpu = deconv.weight.grad
            gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv2d_bias_channels_last(self, dtype=torch.float):
        deconv = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True).to(memory_format=torch.channels_last)

        x_cpu = torch.randn(2, 16, 128, 128, requires_grad=True).to(memory_format=torch.channels_last)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 128, 128).to(memory_format=torch.channels_last)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to('xpu')

        x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to('xpu')
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertTrue(y_xpu.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(gw_xpu.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_group_deconv2d_bias(self, dtype=torch.float):
        deconv = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1, groups=2, bias=True)

        x_cpu = torch.randn(2, 16, 128, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 128, 128)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to('xpu')

        x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to('xpu')
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_group_deconv2d_bias_blk(self, dtype=torch.float):
        deconv = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1, groups=2, bias=True)

        x_cpu = torch.randn(2, 16, 128, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 128, 128)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        with torch.xpu.onednn_layout():
            deconv.zero_grad()
            deconv = deconv.to('xpu')

            x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
            x_xpu.retain_grad()
            gy_xpu = gy_cpu.to('xpu')
            y_xpu = deconv(x_xpu)
            # FIXME:
            # crash with current oneDNN, Floating point exception (core dumped)
            # and will be fixed after oneDNN upgraded to internal master branch with commit df0b87c2e14
            # y_xpu.backward(gy_xpu)
            # gw_xpu = deconv.weight.grad
            # gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        # self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        # self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        # self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_group_deconv2d_bias_channels_last(self, dtype=torch.float):
        deconv = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1, groups=2, bias=True).to(memory_format=torch.channels_last)

        x_cpu = torch.randn(2, 16, 128, 128, requires_grad=True).to(memory_format=torch.channels_last)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 128, 128).to(memory_format=torch.channels_last)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to('xpu')

        x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to('xpu')
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertTrue(y_xpu.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(gw_xpu.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv3d_bias(self, dtype=torch.float):
        deconv = nn.ConvTranspose3d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)

        x_cpu = torch.randn(2, 16, 10, 128, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 10, 128, 128)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to('xpu')

        x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to('xpu')
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv3d_bias_blk(self, dtype=torch.float):
        deconv = nn.ConvTranspose3d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)

        x_cpu = torch.randn(2, 16, 10, 128, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 10, 128, 128)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        with torch.xpu.onednn_layout():
            deconv.zero_grad()
            deconv = deconv.to('xpu')

            x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
            x_xpu.retain_grad()
            gy_xpu = gy_cpu.to('xpu')
            y_xpu = deconv(x_xpu)
            y_xpu.backward(gy_xpu)
            gw_xpu = deconv.weight.grad
            gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv3d_bias_channels_last(self, dtype=torch.float):
        deconv = nn.ConvTranspose3d(16, 32, kernel_size=3, stride=1, padding=1, bias=True).to(memory_format=torch.channels_last_3d)

        x_cpu = torch.randn(2, 16, 10, 128, 128, requires_grad=True).to(memory_format=torch.channels_last_3d)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 10, 128, 128).to(memory_format=torch.channels_last_3d)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to('xpu')

        x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to('xpu')
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertTrue(y_xpu.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertTrue(gw_xpu.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_group_deconv3d_bias(self, dtype=torch.float):
        deconv = nn.ConvTranspose3d(16, 32, kernel_size=3, stride=1, padding=1, groups=2, bias=True)

        x_cpu = torch.randn(2, 16, 10, 128, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 10, 128, 128)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to('xpu')

        x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to('xpu')
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_group_deconv3d_bias_blk(self, dtype=torch.float):
        deconv = nn.ConvTranspose3d(16, 32, kernel_size=3, stride=1, padding=1, groups=2, bias=True)

        x_cpu = torch.randn(2, 16, 10, 128, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 10, 128, 128)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        with torch.xpu.onednn_layout():
            deconv.zero_grad()
            deconv = deconv.to('xpu')

            x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
            x_xpu.retain_grad()
            gy_xpu = gy_cpu.to('xpu')
            y_xpu = deconv(x_xpu)
            # FIXME:
            # crash with current oneDNN, Floating point exception (core dumped)
            # and will be fixed after oneDNN upgraded to internal master branch with commit df0b87c2e14
            # y_xpu.backward(gy_xpu)
            # gw_xpu = deconv.weight.grad
            # gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        # self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        # self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        # self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_group_deconv3d_bias_channels_last(self, dtype=torch.float):
        deconv = nn.ConvTranspose3d(16, 32, kernel_size=3, stride=1, padding=1, groups=2, bias=True).to(memory_format=torch.channels_last_3d)

        x_cpu = torch.randn(2, 16, 10, 128, 128, requires_grad=True).to(memory_format=torch.channels_last_3d)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 10, 128, 128).to(memory_format=torch.channels_last_3d)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to('xpu')

        x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to('xpu')
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertTrue(y_xpu.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertTrue(gw_xpu.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv_no_bias(self, dtype=torch.float):
        deconv = nn.ConvTranspose3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)

        x_cpu = torch.randn(2, 16, 10, 128, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 10, 128, 128)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to('xpu')

        x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to('xpu')
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv_bias_dilation(self, dtype=torch.float):
        deconv = nn.ConvTranspose3d(16, 32, kernel_size=3, stride=1, padding=1, bias=True, dilation=3)

        x_cpu = torch.randn(2, 16, 10, 128, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 14, 132, 132)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to('xpu')

        x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to('xpu')
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv_dilation(self, dtype=torch.float):
        deconv = nn.ConvTranspose3d(16, 32, kernel_size=3, stride=1, padding=1, bias=False, dilation=3)

        x_cpu = torch.randn(2, 16, 10, 128, 128, requires_grad=True)
        x_cpu.retain_grad()
        gy_cpu = torch.randn(2, 32, 14, 132, 132)
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to('xpu')

        x_xpu = x_cpu.detach().clone().to('xpu').requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to('xpu')
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad

        self.assertEqual(y_cpu, y_xpu.cpu())
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-1)

