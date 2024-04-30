import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_deconv3d_1(self, dtype=torch.float):
        shape = ((1, 128, 56, 56, 40), (128, 64, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=4e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=4e-2)

    def test_deconv3d_bias_blk_1(self, dtype=torch.float):
        shape = ((1, 128, 56, 56, 40), (128, 64, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        with torch.xpu.onednn_layout():
            deconv.zero_grad()
            deconv = deconv.to("xpu")

            x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
            x_xpu.retain_grad()
            gy_xpu = gy_cpu.to("xpu")
            y_xpu = deconv(x_xpu)
            y_xpu.backward(gy_xpu)
            gw_xpu = deconv.weight.grad
            gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv3d_bias_channels_last_1(self, dtype=torch.float):
        shape = ((1, 128, 56, 56, 40), (128, 64, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        ).to(memory_format=torch.channels_last_3d)

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device).to(
            memory_format=torch.channels_last_3d
        )
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        ).to(
            memory_format=torch.channels_last_3d
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertTrue(y_xpu.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertTrue(gw_xpu.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv_no_bias_1(self, dtype=torch.float):
        shape = ((1, 128, 56, 56, 40), (128, 64, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv3d_2(self, dtype=torch.float):
        shape = ((1, 256, 28, 28, 20), (256, 128, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=4e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=4e-2)

    def test_deconv3d_bias_blk_2(self, dtype=torch.float):
        shape = ((1, 256, 28, 28, 20), (256, 128, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        with torch.xpu.onednn_layout():
            deconv.zero_grad()
            deconv = deconv.to("xpu")

            x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
            x_xpu.retain_grad()
            gy_xpu = gy_cpu.to("xpu")
            y_xpu = deconv(x_xpu)
            y_xpu.backward(gy_xpu)
            gw_xpu = deconv.weight.grad
            gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv3d_bias_channels_last_2(self, dtype=torch.float):
        shape = ((1, 256, 28, 28, 20), (256, 128, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        ).to(memory_format=torch.channels_last_3d)

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device).to(
            memory_format=torch.channels_last_3d
        )
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        ).to(
            memory_format=torch.channels_last_3d
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertTrue(y_xpu.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertTrue(gw_xpu.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv_no_bias_2(self, dtype=torch.float):
        shape = ((1, 256, 28, 28, 20), (256, 128, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv3d_3(self, dtype=torch.float):
        shape = ((1, 320, 14, 14, 10), (320, 256, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=4e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=4e-2)

    def test_deconv3d_bias_blk_3(self, dtype=torch.float):
        shape = ((1, 320, 14, 14, 10), (320, 256, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        with torch.xpu.onednn_layout():
            deconv.zero_grad()
            deconv = deconv.to("xpu")

            x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
            x_xpu.retain_grad()
            gy_xpu = gy_cpu.to("xpu")
            y_xpu = deconv(x_xpu)
            y_xpu.backward(gy_xpu)
            gw_xpu = deconv.weight.grad
            gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv3d_bias_channels_last_3(self, dtype=torch.float):
        shape = ((1, 320, 14, 14, 10), (320, 256, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        ).to(memory_format=torch.channels_last_3d)

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device).to(
            memory_format=torch.channels_last_3d
        )
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        ).to(
            memory_format=torch.channels_last_3d
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertTrue(y_xpu.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertTrue(gw_xpu.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv_no_bias_3(self, dtype=torch.float):
        shape = ((1, 320, 14, 14, 10), (320, 256, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv3d_4(self, dtype=torch.float):
        shape = ((1, 320, 7, 7, 5), (320, 320, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=4e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=4e-2)

    def test_deconv3d_bias_blk_4(self, dtype=torch.float):
        shape = ((1, 320, 7, 7, 5), (320, 320, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        with torch.xpu.onednn_layout():
            deconv.zero_grad()
            deconv = deconv.to("xpu")

            x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
            x_xpu.retain_grad()
            gy_xpu = gy_cpu.to("xpu")
            y_xpu = deconv(x_xpu)
            y_xpu.backward(gy_xpu)
            gw_xpu = deconv.weight.grad
            gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv3d_bias_channels_last_4(self, dtype=torch.float):
        shape = ((1, 320, 7, 7, 5), (320, 320, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        ).to(memory_format=torch.channels_last_3d)

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device).to(
            memory_format=torch.channels_last_3d
        )
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        ).to(
            memory_format=torch.channels_last_3d
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertTrue(y_xpu.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertTrue(gw_xpu.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv_no_bias_4(self, dtype=torch.float):
        shape = ((1, 320, 7, 7, 5), (320, 320, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv3d_5(self, dtype=torch.float):
        shape = ((1, 64, 112, 112, 80), (64, 32, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=4e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=4e-2)

    def test_deconv3d_bias_blk_5(self, dtype=torch.float):
        shape = ((1, 64, 112, 112, 80), (64, 32, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        with torch.xpu.onednn_layout():
            deconv.zero_grad()
            deconv = deconv.to("xpu")

            x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
            x_xpu.retain_grad()
            gy_xpu = gy_cpu.to("xpu")
            y_xpu = deconv(x_xpu)
            y_xpu.backward(gy_xpu)
            gw_xpu = deconv.weight.grad
            gb_xpu = deconv.bias.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv3d_bias_channels_last_5(self, dtype=torch.float):
        shape = ((1, 64, 112, 112, 80), (64, 32, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        ).to(memory_format=torch.channels_last_3d)

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device).to(
            memory_format=torch.channels_last_3d
        )
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        ).to(
            memory_format=torch.channels_last_3d
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()
        gb_cpu = deconv.bias.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad
        gb_xpu = deconv.bias.grad

        self.assertTrue(y_xpu.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertTrue(gw_xpu.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
        self.assertEqual(gb_cpu, gb_xpu.cpu(), rtol=1e-3, atol=1e-2)

    def test_deconv_no_bias_5(self, dtype=torch.float):
        shape = ((1, 64, 112, 112, 80), (64, 32, 2, 2, 2))
        N, C, D, H, W = shape[0][0], shape[0][1], shape[0][2], shape[0][3], shape[0][4]
        deconv = nn.ConvTranspose3d(
            shape[1][0], shape[1][1], kernel_size=(shape[1][2], shape[1][3], shape[1][4]), stride=2, padding=1, bias=True, dtype=dtype
        )

        x_cpu = torch.randn([N, C, D, H, W], requires_grad=True, dtype=dtype, device=cpu_device)
        x_cpu.retain_grad()
        gy_cpu = torch.full(
            [N, shape[1][1], (D-1)*2-2+(shape[1][2]-1) + 1, (H-1)*2-2+(shape[1][3]-1) + 1, (W-1)*2-2+(shape[1][4]-1) + 1], 1e-3, dtype=dtype, device=cpu_device, requires_grad=True
        )
        y_cpu = deconv(x_cpu)
        y_cpu.backward(gy_cpu)
        gw_cpu = deconv.weight.grad.detach().clone()

        deconv.zero_grad()
        deconv = deconv.to("xpu")

        x_xpu = x_cpu.detach().clone().to("xpu").requires_grad_()
        x_xpu.retain_grad()
        gy_xpu = gy_cpu.to("xpu")
        y_xpu = deconv(x_xpu)
        y_xpu.backward(gy_xpu)
        gw_xpu = deconv.weight.grad

        self.assertEqual(y_cpu, y_xpu.cpu(), rtol=1e-5, atol=1e-5)
        self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())
        self.assertEqual(gw_cpu, gw_xpu.cpu(), rtol=1e-3, atol=1e-2)
