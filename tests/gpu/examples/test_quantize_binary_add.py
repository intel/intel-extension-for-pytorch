import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import os
import copy
import pytest

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestNNMethod(TestCase):
    def test_qbinary(self, dtype=torch.float):

        x_cpu = torch.randn([1, 1, 3, 3], device=cpu_device)
        y_cpu = torch.randn([1, 1, 3, 3], device=cpu_device)

        zero_point = 0
        scale_in_1 = 0.421009
        scale_in_2 = 2.04386

        scale_out = 0.2

        dtype_inputs = torch.qint8

        q_x_cpu = torch.quantize_per_tensor(x_cpu, scale_in_1, zero_point, dtype_inputs)
        q_y_cpu = torch.quantize_per_tensor(y_cpu, scale_in_2, zero_point, dtype_inputs)

        x_xpu = x_cpu.to("xpu")
        y_xpu = y_cpu.to("xpu")

        q_x_xpu = torch.quantize_per_tensor(x_xpu, scale_in_1, zero_point, dtype_inputs)
        q_y_xpu = torch.quantize_per_tensor(y_xpu, scale_in_2, zero_point, dtype_inputs)

        ref1 = torch.ops.quantized.add(q_x_cpu, q_y_cpu, scale_out, 0)
        real1 = torch.ops.quantized.add(q_x_xpu, q_y_xpu, scale_out, 0)
        
        self.assertEqual(ref1, real1)
