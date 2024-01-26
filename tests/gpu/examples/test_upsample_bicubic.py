import torch
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


class TestNNMethod(TestCase):
    def test_upsample_bicubic_2d(self, dtype=torch.float):
        input_cpu = torch.randn((1, 3, 5, 5), dtype=torch.float32)
        input_xpu = input_cpu.to("xpu")
        scales = 1
        input_cpu.requires_grad = True
        input_xpu.requires_grad = True
        alc = False
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu,
            scale_factor=scales,
            mode="bicubic",
            align_corners=alc,
            recompute_scale_factor=rsf,
        )
        output_xpu = torch.nn.functional.interpolate(
            input_xpu,
            scale_factor=scales,
            mode="bicubic",
            align_corners=alc,
            recompute_scale_factor=rsf,
        )
        self.assertEqual(output_cpu, output_xpu.cpu())

        grad_out_cpu = torch.randn((1, 3, 5, 5), dtype=torch.float32)
        grad_out_xpu = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_xpu = Variable(grad_out_xpu, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_xpu.backward(grad_out_xpu)
        grad_cpu = input_cpu.grad
        grad_xpu = input_xpu.grad
        self.assertEqual(grad_cpu, grad_xpu.cpu())

    def test_test_upsample_bicubic_2d_half(self, dtype=torch.float):
        input_cpu = torch.randn((1, 3, 5, 5), dtype=torch.float32)
        input_xpu = input_cpu.to("xpu", dtype=torch.float16)
        scales = 1
        input_cpu.requires_grad = True
        input_xpu.requires_grad = True
        alc = False
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu,
            scale_factor=scales,
            mode="bicubic",
            align_corners=alc,
            recompute_scale_factor=rsf,
        )
        output_xpu = torch.nn.functional.interpolate(
            input_xpu,
            scale_factor=scales,
            mode="bicubic",
            align_corners=alc,
            recompute_scale_factor=rsf,
        )
        self.assertEqual(
            output_cpu, output_xpu.to("cpu", dtype=torch.float32), atol=1e-3, rtol=1e-3
        )

        grad_out_cpu = torch.randn((1, 3, 5, 5), dtype=torch.float32)
        grad_out_xpu = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_xpu = Variable(grad_out_xpu, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_xpu.backward(grad_out_xpu)
        grad_cpu = input_cpu.grad
        grad_xpu = input_xpu.grad
        self.assertEqual(
            grad_cpu, grad_xpu.to("cpu", dtype=torch.float32), atol=1e-3, rtol=1e-3
        )

    def test_upsample_bicubic_2d_align_corners(self, dtype=torch.float):
        input_cpu = torch.randn((1, 3, 5, 5), dtype=torch.float32)
        input_xpu = input_cpu.to("xpu")
        scales = 1
        input_cpu.requires_grad = True
        input_xpu.requires_grad = True
        alc = True  # align corners
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu,
            scale_factor=scales,
            mode="bicubic",
            align_corners=alc,
            recompute_scale_factor=rsf,
        )
        output_xpu = torch.nn.functional.interpolate(
            input_xpu,
            scale_factor=scales,
            mode="bicubic",
            align_corners=alc,
            recompute_scale_factor=rsf,
        )
        self.assertEqual(output_cpu, output_xpu.cpu())

        grad_out_cpu = torch.randn((1, 3, 5, 5), dtype=torch.float32)
        grad_out_xpu = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_xpu = Variable(grad_out_xpu, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_xpu.backward(grad_out_xpu)
        grad_cpu = input_cpu.grad
        grad_xpu = input_xpu.grad
        self.assertEqual(grad_cpu, grad_xpu.cpu())
