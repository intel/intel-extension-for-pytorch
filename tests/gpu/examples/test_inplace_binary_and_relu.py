import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch # noqa

"""
    Motivation: When running block format ResNet-XX with using inplaced binary add and relu, 
                layer output will be flushed to shape zero. In this UT, it may throw a segmentfault error.
    This issue was fixed in commit: 193ef42, 
    PR: https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu/pull/668/commits
"""

import pytest

dpcpp_device = torch.device("xpu")
cpu_device = torch.device("cpu")


class TestTorchMethod(TestCase):

    def test_inplace_binary_and_relu(self, dtype=torch.float):
        with torch.xpu.onednn_layout():
            input_cpu = torch.randn([32, 64, 300, 300])

            conv_cpu = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
            relu_cpu = nn.ReLU(inplace=True)

            out_cpu = conv_cpu(input_cpu)
            out_cpu += input_cpu
            out_cpu = relu_cpu(out_cpu)
            out_cpu_shape = out_cpu.shape
            print("\n")
            print(out_cpu_shape)

            input_dpcpp = input_cpu.to(dpcpp_device)

            conv_dpcpp = conv_cpu.to(dpcpp_device)
            relu_dpcpp = relu_cpu.to(dpcpp_device)

            out_dpcpp = conv_dpcpp(input_dpcpp)
            out_dpcpp += input_dpcpp
            out_dpcpp = relu_dpcpp(out_dpcpp)
            out_dpcpp_shape = out_dpcpp.shape
            print(out_dpcpp_shape)
            self.assertEqual(out_cpu_shape, out_dpcpp_shape)
