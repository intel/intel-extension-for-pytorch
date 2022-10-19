import torch
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestNNMethod(TestCase):
    def test_upsamle_linear_1d(self, dtype=torch.float):
        input_cpu = torch.randn((2, 3, 5), dtype=torch.float32)
        input_dpcpp = input_cpu.to("xpu")
        scales = [6]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = False  # we don't support this path by currently as oneDNN don't support this algorithm!
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode='linear', align_corners=alc, recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='linear', align_corners=alc, recompute_scale_factor=rsf)
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.randn((2, 3, 30), dtype=torch.float32)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

    def test_upsamle_linear_2d(self, dtype=torch.float):
        input_cpu = torch.randn((2, 3, 5, 5), dtype=torch.float32)
        input_dpcpp = input_cpu.to("xpu")
        scales = [6, 8]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = False
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode='bilinear', align_corners=alc, recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(input_dpcpp, scale_factor=scales, mode='bilinear',
                                                       align_corners=alc, recompute_scale_factor=rsf)
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.randn((2, 3, 30, 40), dtype=torch.float32)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_upsamle_linear_2d_align_corners(self, dtype=torch.float):
        input_cpu = torch.randn((2, 3, 5, 5), dtype=torch.float64)
        input_dpcpp = input_cpu.to("xpu")
        scales = [6, 8]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = True  # align corners
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode='bilinear', align_corners=alc, recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(input_dpcpp, scale_factor=scales, mode='bilinear',
                                                       align_corners=alc, recompute_scale_factor=rsf)
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.randn((2, 3, 30, 40), dtype=torch.float32)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

    def test_upsamle_linear_3d(self, dtype=torch.float):
        input_cpu = torch.randn((2, 3, 2, 5, 5), dtype=torch.float32)
        input_dpcpp = input_cpu.to("xpu")
        scales = [6, 8, 1]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        alc = False
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode='trilinear', align_corners=alc, recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(input_dpcpp, scale_factor=scales, mode='trilinear',
                                                       align_corners=alc, recompute_scale_factor=rsf)
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.randn((2, 3, 12, 40, 5), dtype=torch.float32)
        grad_out_dpcpp = grad_out_cpu.to("xpu")
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())
