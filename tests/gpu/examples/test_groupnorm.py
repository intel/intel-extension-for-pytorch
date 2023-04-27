import torch
import intel_extension_for_pytorch # noqa
from torch.testing._internal.common_utils import TestCase
import torch.nn as nn

class TestNet(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, affine=True):
        super(TestNet, self).__init__()
        self.norm = nn.GroupNorm(8, 16, affine=affine)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False)
        self.s = nn.Sequential(self.conv, self.norm)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class TestTorchMethod(TestCase):
    def test_group_norm_forward(self, dtype=torch.float):
        # x = torch.randn(32,64, 540, 960, requires_grad=True, device=torch.device('cpu'))
        x = torch.randn(16, 16, 64, 64, requires_grad=True, device=torch.device('cpu'))
        x_dpcpp = x.to("xpu")
        # y = torch.randn(32, 64, 538, 958, requires_grad=True, device=torch.device('xpu'))

        model = TestNet(16, 16)

        model.eval()
        y_pred = model(x)
        model_dpcpp = model.to(torch.device('xpu'))
        model_dpcpp.eval()
        y_pred_dpcpp = model_dpcpp(x_dpcpp)
        # print(y_pred)
        # print(y_pred_dpcpp.to("xpu"))
        self.assertEqual(y_pred_dpcpp.is_contiguous(), True)
        self.assertEqual(y_pred, y_pred_dpcpp)

    def test_group_norm_forward_no_gamma_no_beta(self, dtype=torch.float):
        x = torch.randn(16, 16, 64, 64, requires_grad=True, device=torch.device('cpu'))
        x_dpcpp = x.to("xpu")

        model = TestNet(16, 16, 1, False)

        model.eval()
        y_pred = model(x)
        model_dpcpp = model.to(torch.device('xpu'))
        model_dpcpp.eval()
        y_pred_dpcpp = model_dpcpp(x_dpcpp)
        self.assertEqual(y_pred_dpcpp.is_contiguous(), True)
        self.assertEqual(y_pred, y_pred_dpcpp)

    def test_group_norm_backward(self, dtype=torch.float):
        x = torch.randn(16, 16, 64, 64, requires_grad=True, device=torch.device('cpu'))
        x_dpcpp = x.to("xpu")

        for model in [TestNet(16, 16), nn.GroupNorm(8, 16, affine=True)]:
            model.eval()
            y_pred = model(x)
            grad_out = torch.randn_like(y_pred)
            grad, = torch.autograd.grad(y_pred, x, grad_out)

            model_dpcpp = model.to(torch.device('xpu'))
            model_dpcpp.eval()
            y_pred_dpcpp = model_dpcpp(x_dpcpp)
            grad_dpcpp, = torch.autograd.grad(y_pred_dpcpp, x_dpcpp, grad_out.to("xpu"))
            self.assertEqual(grad_dpcpp.is_contiguous(), True)
            self.assertEqual(grad, grad_dpcpp)

    def test_group_norm_backward_autocast(self, dtype=torch.float):
        x = torch.randn(16, 16, 64, 64, requires_grad=True, device=torch.device('cpu'))
        x_dpcpp = x.to("xpu")

        for model in [TestNet(16, 16), nn.GroupNorm(8, 16, affine=True)]:
            model.eval()
            y_pred = model(x)
            grad_out = torch.randn_like(y_pred)
            grad, = torch.autograd.grad(y_pred, x, grad_out)

            model_dpcpp = model.to(torch.device('xpu'))
            model_dpcpp.eval()
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                y_pred_dpcpp = model_dpcpp(x_dpcpp)
            grad_dpcpp, = torch.autograd.grad(y_pred_dpcpp, x_dpcpp, grad_out.to("xpu"))
            self.assertEqual(grad_dpcpp.is_contiguous(), True)
            self.assertEqual(grad, grad_dpcpp, atol=1e-1, rtol=1e-1)

    def test_group_norm_backward_no_gamma_no_beta(self, dtype=torch.float):
        x = torch.randn(16, 16, 64, 64, requires_grad=True, device=torch.device('cpu'))
        x_dpcpp = x.to("xpu")

        for model in [TestNet(16, 16, 1, False), nn.GroupNorm(8, 16, affine=False)]:
            model.eval()
            y_pred = model(x)
            grad_out = torch.randn_like(y_pred)
            grad, = torch.autograd.grad(y_pred, x, grad_out)

            model_dpcpp = model.to(torch.device('xpu'))
            model_dpcpp.eval()
            y_pred_dpcpp = model_dpcpp(x_dpcpp)
            grad_dpcpp, = torch.autograd.grad(y_pred_dpcpp, x_dpcpp, grad_out.to("xpu"))
            self.assertEqual(grad_dpcpp.is_contiguous(), True)
            self.assertEqual(grad, grad_dpcpp)

    def test_group_norm(self):
        shapes = [[2, 2560, 32, 32],
                [2, 2560, 16, 16],
                [2, 2560, 8, 8],
                [2, 1920, 32, 32],
                [2, 1920, 16, 16],
                [2, 1920, 8, 8],
                [2, 1280, 32, 32],
                [2, 1280, 16, 16],
                [2, 1280, 8, 8],
                [2, 960, 64, 64],
                [2, 960, 32, 32],
                [2, 960, 16, 16],
                [2, 640, 64, 64],
                [2, 640, 32, 32],
                [2, 640, 16, 16],
                [2, 320, 64, 64],
                [1, 512, 128, 128],
                [1, 512, 64, 64],
                [1, 256, 256, 256],
                [1, 128, 512, 512],
                [1, 256, 513, 513],
                [1, 128, 512, 512],
                [1, 256, 55, 55],
                [1, 128, 7, 7]]
        groups = [128, 32]
        formats = [torch.contiguous_format, torch.channels_last]
        dtypes = [torch.float]

        for shape in shapes:
            for group in groups:
                for format in formats:
                    for dtype in dtypes:
                        group = min(group, shape[1])
                        if (shape[1] % group):
                            continue

                        input = torch.randn(shape)
                        grad = torch.randn(shape)
                        input = input.to(memory_format=format)
                        grad = grad.to(memory_format=format)

                        input_cpu = input.clone()
                        input_cpu.requires_grad_(True)
                        grad_cpu = grad.clone()
                        m = torch.nn.GroupNorm(group, shape[1])
                        output_cpu = m(input_cpu)
                        output_cpu.backward(grad_cpu)
                        grad_wei = m.weight.grad.clone()

                        input_xpu = input.clone().to("xpu").to(dtype)
                        input_xpu.requires_grad_(True)
                        grad_xpu = grad.clone().to("xpu")
                        model_xpu = m.to("xpu").to(dtype)
                        model_xpu.zero_grad()
                        output_xpu = model_xpu(input_xpu)
                        output_xpu.backward(grad_xpu)
                        grad_wei_xpu = model_xpu.weight.grad.clone()

                        self.assertEqual(output_cpu.float(), output_xpu.cpu().float())
                        self.assertEqual(input_cpu.grad.float(), input_xpu.grad.cpu().float())
                        self.assertEqual(grad_wei, grad_wei_xpu, atol=7e-3, rtol=7e-3)
