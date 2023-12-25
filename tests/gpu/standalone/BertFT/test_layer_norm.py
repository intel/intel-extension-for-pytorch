from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_layer_norm_half(self, dtype=torch.half):
        input_shapes = [
            [96, 384, 1024],
            [2, 384, 1024]
        ]
        norm_shapes = [
            [1024],
            [1024],
            [36864],
            [36864],
        ]
        for idx, input_shape in enumerate(input_shapes):
            x_i = torch.randn(input_shape, device=cpu_device)
            x_dpcpp_i = x_i.to(dpcpp_device).to(dtype)
            norm_shape = norm_shapes[idx]
            layernorm = nn.LayerNorm(norm_shape)
            y_cpu = layernorm(x_i)
            layernorm.to(dpcpp_device).to(dtype)
            y_dpcpp = layernorm(x_dpcpp_i)
            self.assertEqual(y_cpu, y_dpcpp.float(), atol=1e-1, rtol=0)

    def test_layer_norm_bfloat16(self, dtype=torch.bfloat16):
        input_shapes = [
            [96, 384, 1024],
            [2, 384, 1024]
        ]
        norm_shapes = [
            [1024],
            [1024],
            [36864],
            [36864],
        ]
        for idx, input_shape in enumerate(input_shapes):
            x_i = torch.randn(input_shape, device=cpu_device)
            x_dpcpp_i = x_i.to(dpcpp_device).to(dtype)
            norm_shape = norm_shapes[idx]
            layernorm = nn.LayerNorm(norm_shape)
            y_cpu = layernorm(x_i)
            layernorm.to(dpcpp_device).to(dtype)
            y_dpcpp = layernorm(x_dpcpp_i)
            self.assertEqual(y_cpu, y_dpcpp.float(), atol=1e-1, rtol=0)

    def test_layer_norm_fwd_bwd(self, dtype=torch.float):
        formats = [torch.contiguous_format, torch.channels_last]
        input_shapes = [
            [96, 384, 1024],
            [2, 384, 1024]
        ]
        norm_shapes = [
            [1024],
            [1024],
            [36864],
            [36864],
        ]

        for idx, input_shape in enumerate(input_shapes):
            for format in formats:
                # One of many seeds for which this UT passes for Arc GPUs on machines with consumer-grade CPUs, which
                # do not have as many cores as Xeon servers, and the difference between the GPU and the CPU output
                # may exceed tolerance bounds, at times, because layernorm CPU ATen kernels would produce divergent
                # outputs on a consumer-grade CPU and a Xeon server, since this UT uses more threads on a typical Xeon
                # machine, than on a typical consumer-grade machine with Arc, affecting the output simply due to the
                # nature of float arithmetic. Another reason for divergence is that Xeon machines have FMA units,
                # which produce more accurate output than consumer-grade machines that lack FMA units.
                torch.manual_seed(13)
                norm_shape = norm_shapes[idx]
                input = torch.randn(input_shape)
                grad = torch.randn(input_shape)
                if input.dim() == 4:
                    input = input.to(memory_format=format)
                    grad = grad.to(memory_format=format)

                input_cpu = input.clone()
                input_cpu.requires_grad_(True)
                grad_cpu = grad.clone()
                m = torch.nn.LayerNorm(norm_shape)
                output_cpu = m(input_cpu)
                output_cpu.backward(grad_cpu)
                grad_wei = m.weight.grad.clone()

                m.zero_grad()
                input_xpu = input.clone().to("xpu").to(dtype)
                input_xpu.requires_grad_(True)
                grad_xpu = grad.clone().to("xpu")
                model_xpu = m.to("xpu").to(dtype)
                model_xpu.zero_grad()
                output_xpu = model_xpu(input_xpu)
                output_xpu.backward(grad_xpu)
                grad_wei_xpu = model_xpu.weight.grad.clone()

                self.assertEqual(output_cpu, output_xpu.cpu())
                self.assertEqual(input_cpu.grad, input_xpu.grad.cpu())
                self.assertEqual(grad_wei, grad_wei_xpu.cpu(), rtol=10e-4, atol=10e-4)
