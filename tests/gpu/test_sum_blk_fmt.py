import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import os
import pytest


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

os.environ["IPEX_LAZY_REORDER"] = "1"
os.environ["IPEX_WEIGHT_CACHE"] = "1"


class TestNNMethod(TestCase):
    def test_conv_relu_fusion(self, dtype=torch.float):
        conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)
        conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)
        x_cpu = torch.randn([1, 16, 3, 3], device=cpu_device)
        y_cpu = torch.randn([1, 16, 3, 3], device=cpu_device)
        ref1 = conv1(x_cpu) + conv2(y_cpu)
        ref2 = x_cpu + conv2(y_cpu)
        ref3 = conv1(x_cpu) + y_cpu

        conv1_dpcpp = conv1.to("dpcpp")
        conv2_dpcpp = conv2.to("dpcpp")
        x_dpcpp = x_cpu.to("dpcpp")
        y_dpcpp = y_cpu.to("dpcpp")
        real1 = conv1_dpcpp(x_dpcpp) + conv2_dpcpp(y_dpcpp)
        real2 = x_dpcpp + conv2_dpcpp(y_dpcpp)
        real3 = conv1_dpcpp(x_dpcpp) + y_dpcpp

        self.assertEqual(ref1, real1.to(cpu_device))
        self.assertEqual(ref2, real2.to(cpu_device))
        self.assertEqual(ref3, real3.to(cpu_device))
