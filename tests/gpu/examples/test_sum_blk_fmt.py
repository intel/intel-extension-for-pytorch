import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import ipex

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_sum_blk_fusion(self, dtype=torch.float):
        conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)
        conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)
        x_cpu = torch.randn([1, 16, 3, 3], device=cpu_device)
        y_cpu = torch.randn([1, 16, 3, 3], device=cpu_device)
        ref1 = conv1(x_cpu) + conv2(y_cpu)
        ref2 = x_cpu + conv2(y_cpu)
        ref3 = conv1(x_cpu) + y_cpu
        ref4 = conv1(x_cpu) + 2

        conv1_dpcpp = conv1.to("xpu")
        conv2_dpcpp = conv2.to("xpu")
        x_dpcpp = x_cpu.to("xpu")
        y_dpcpp = y_cpu.to("xpu")
        real1 = conv1_dpcpp(x_dpcpp) + conv2_dpcpp(y_dpcpp)
        real2 = x_dpcpp + conv2_dpcpp(y_dpcpp)
        real3 = conv1_dpcpp(x_dpcpp) + y_dpcpp
        real4 = conv1_dpcpp(x_dpcpp) + 2

        self.assertEqual(ref1, real1.to(cpu_device))
        self.assertEqual(ref2, real2.to(cpu_device))
        self.assertEqual(ref3, real3.to(cpu_device))
        self.assertEqual(ref4, real4.to(cpu_device))
