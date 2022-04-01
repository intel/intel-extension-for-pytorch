import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch
import pytest

cpu_device = torch.device("cpu")
sycl_device = torch.device("xpu")


class TestNNMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.has_channels_last_1d() or torch.xpu.using_layout_opt(), reason="doesn't enable channels last 1d or channels last does not support onednn block format")
    def test_upsamle_nearest_channels_last_1d(self, dtype=torch.float):
        # #### upsample nearest 1D #####
        input_cpu = torch.randn((2, 3, 5), dtype=torch.float32, device=cpu_device).to(memory_format=torch.channels_last_1d)
        input_dpcpp = input_cpu.to("xpu").to(memory_format=torch.channels_last_1d)
        scales = [6]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.randn((2, 3, 30), dtype=torch.float32, device=cpu_device).to(memory_format=torch.channels_last_1d)
        grad_out_dpcpp = grad_out_cpu.to("xpu").to(memory_format=torch.channels_last_1d)
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

    @pytest.mark.skipif(torch.xpu.using_layout_opt(), reason="channels last does not support onednn block format")
    def test_upsamle_nearest_channels_last(self, dtype=torch.float):
        # #### upsample nearest 2D #####
        input_cpu = torch.randn((2, 3, 5, 5), dtype=torch.float32, device=cpu_device).to(memory_format=torch.channels_last)
        input_dpcpp = input_cpu.to("xpu").to(memory_format=torch.channels_last)
        scales = [6, 8]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.randn((2, 3, 30, 40), dtype=torch.float32, device=cpu_device).to(memory_format=torch.channels_last)
        grad_out_dpcpp = grad_out_cpu.to("xpu").to(memory_format=torch.channels_last)
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())

    @pytest.mark.skipif(torch.xpu.using_layout_opt(), reason="channels last does not support onednn block format")
    def test_upsamle_nearest_channels_last_3d(self, dtype=torch.float):
        # #### upsample nearest 3D #####
        input_cpu = torch.randn((2, 3, 2, 5, 5), dtype=torch.float32, device=cpu_device).to(memory_format=torch.channels_last_3d)
        input_dpcpp = input_cpu.to("xpu").to(memory_format=torch.channels_last_3d)
        scales = [6, 8, 1]
        input_cpu.requires_grad = True
        input_dpcpp.requires_grad = True
        rsf = False

        output_cpu = torch.nn.functional.interpolate(
            input_cpu, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        output_dpcpp = torch.nn.functional.interpolate(
            input_dpcpp, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        self.assertEqual(output_cpu, output_dpcpp.cpu())

        grad_out_cpu = torch.randn((2, 3, 12, 40, 5), dtype=torch.float32, device=cpu_device).to(memory_format=torch.channels_last_3d)
        grad_out_dpcpp = grad_out_cpu.to("xpu").to(memory_format=torch.channels_last_3d)
        grad_out_cpu = Variable(grad_out_cpu, requires_grad=True)
        grad_out_dpcpp = Variable(grad_out_dpcpp, requires_grad=True)

        output_cpu.backward(grad_out_cpu)
        output_dpcpp.backward(grad_out_dpcpp)
        grad_cpu = input_cpu.grad
        grad_dpcpp = input_dpcpp.grad
        self.assertEqual(grad_cpu, grad_dpcpp.cpu())
