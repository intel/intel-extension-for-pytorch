import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_instance_norm2d(self):
        test_conv = torch.nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1))
        test_module = torch.nn.InstanceNorm2d(2)
        test_module.weight = torch.nn.Parameter(torch.randn(3))
        test_module.bias = torch.nn.Parameter(torch.randn(3))

        rand_input = torch.randn((1, 3, 7, 5))

        cpu_result = test_conv(rand_input)
        cpu_result = test_module(cpu_result)

        xpu_conv = test_conv.to("xpu")
        xpu_module = test_module.to("xpu")
        xpu_result = xpu_conv(rand_input.to("xpu"))
        xpu_result = xpu_module(xpu_result)
        self.assertEqual(cpu_result, xpu_result.to("cpu"))

    def test_instance_norm3d(self):
        test_conv = torch.nn.Conv3d(3, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        test_module = torch.nn.InstanceNorm3d(3)
        test_module.weight = torch.nn.Parameter(torch.randn(3))
        test_module.bias = torch.nn.Parameter(torch.randn(3))

        rand_input = torch.randn((1, 3, 7, 7, 5))

        cpu_result = test_conv(rand_input)
        cpu_result = test_module(cpu_result)

        xpu_conv = test_conv.to("xpu")
        xpu_module = test_module.to("xpu")
        xpu_result = xpu_conv(rand_input.to("xpu"))
        xpu_result = xpu_module(xpu_result)
        self.assertEqual(cpu_result, xpu_result.to("cpu"))

    @pytest.mark.skipif(torch.xpu.using_layout_opt(), reason="channels last does not support onednn block format")
    def test_instance_norm2d_channels_last(self, dtype=torch.float):
        shapes = [(1, 3, 7, 7), (2, 2, 3, 3), (4, 4, 4, 4), (4, 4, 1, 2), (4, 1, 4, 4),
                  (4, 1, 4, 1), (4, 1, 1, 4), (1, 4, 1, 4), (1, 4, 4, 1), (4, 1, 2, 1)]
        for shape in shapes:
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            test_conv = torch.nn.Conv2d(C, C, kernel_size=(3, 3), padding=(1, 1))
            test_module = torch.nn.InstanceNorm2d(2)
            test_module.weight = torch.nn.Parameter(torch.randn(C))
            test_module.bias = torch.nn.Parameter(torch.randn(C))

            rand_input = torch.randn((N, C, H, W))

            cpu_result = test_conv(rand_input)
            cpu_result = test_module(cpu_result)

            xpu_conv = test_conv.to("xpu")
            xpu_module = test_module.to("xpu")
            xpu_result = xpu_conv(rand_input.to("xpu"))
            xpu_result = xpu_module(xpu_result)
            self.assertEqual(cpu_result, xpu_result.to("cpu"))

            print("-----start channel last----")
            ch_conv = test_conv.to("cpu").to(memory_format=torch.channels_last)
            ch_module = test_module.to("cpu").to(memory_format=torch.channels_last)
            ch_input = rand_input.to("cpu").to(memory_format=torch.channels_last)
            ch_cpu_result = ch_conv(ch_input)
            ch_cpu_result = ch_module(ch_cpu_result)

            ch_xpu_conv = test_conv.to("xpu").to(memory_format=torch.channels_last)
            ch_xpu_module = test_module.to("xpu").to(memory_format=torch.channels_last)
            ch_xpu_input = rand_input.to("xpu").to(memory_format=torch.channels_last)
            ch_xpu_result = ch_xpu_conv(ch_xpu_input)
            ch_xpu_result = ch_xpu_module(ch_xpu_result)
            self.assertEqual(ch_cpu_result, ch_xpu_result.to("cpu"))

    @pytest.mark.skipif(torch.xpu.using_layout_opt(), reason="channels last does not support onednn block format")
    def test_instance_norm3d_channels_last(self, dtype=torch.float):
        shapes = [(1, 3, 7, 7, 5), (3, 3, 7, 7, 5)]
        for shape in shapes:
            N, C, H, W, D = shape[0], shape[1], shape[2], shape[3], shape[4]
            test_conv = torch.nn.Conv3d(C, C, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            test_module = torch.nn.InstanceNorm3d(3)
            test_module.weight = torch.nn.Parameter(torch.randn(C))
            test_module.bias = torch.nn.Parameter(torch.randn(C))

            rand_input = torch.randn((N, C, H, W, D))

            cpu_result = test_conv(rand_input)
            cpu_result = test_module(cpu_result)

            xpu_conv = test_conv.to("xpu")
            xpu_module = test_module.to("xpu")
            xpu_result = xpu_conv(rand_input.to("xpu"))
            xpu_result = xpu_module(xpu_result)
            self.assertEqual(cpu_result, xpu_result.to("cpu"))

            print("-----start channel last----")
            ch_conv = test_conv.to("cpu").to(memory_format=torch.channels_last_3d)
            ch_module = test_module.to("cpu").to(memory_format=torch.channels_last_3d)
            ch_input = rand_input.to("cpu").to(memory_format=torch.channels_last_3d)
            ch_cpu_result = ch_conv(ch_input)
            ch_cpu_result = ch_module(ch_cpu_result)

            ch_xpu_conv = test_conv.to("xpu").to(memory_format=torch.channels_last_3d)
            ch_xpu_module = test_module.to("xpu").to(memory_format=torch.channels_last_3d)
            ch_xpu_input = rand_input.to("xpu").to(memory_format=torch.channels_last_3d)
            ch_xpu_result = ch_xpu_conv(ch_xpu_input)
            ch_xpu_result = ch_xpu_module(ch_xpu_result)
            self.assertEqual(ch_cpu_result, ch_xpu_result.to("cpu"))
