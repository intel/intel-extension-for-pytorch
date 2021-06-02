import torch
import torch_ipex
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

class TestNNMethod(TestCase):
    def test_q_upsamle_nearest(self, dtype=torch.float):
        x_cpu = torch.randn((2,3,5,5), dtype=torch.float32, device = torch.device("cpu"))
        x_gpu = x_cpu.to("xpu")
        scales = [6, 8]
        rsf = False

        dtype_inputs = torch.qint8
        q_scale=0.04
        q_cpu = torch.quantize_per_tensor(x_cpu, q_scale, 0, dtype_inputs)
        q_gpu = torch.quantize_per_tensor(x_gpu, q_scale, 0, dtype_inputs)

        output_cpu = torch.nn.functional.interpolate(q_cpu, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        output_gpu = torch.nn.functional.interpolate(q_gpu, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)

        self.assertEqual(output_cpu, output_gpu)
