import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch
import pytest

class TestNNMethod(TestCase):
    @pytest.mark.skipif(torch.xpu.using_onednn_layout(), reason="channels last does not support onednn block format")
    def test_q_upsamle_nearest_channels_last(self, dtype=torch.float):
        x_cpu = torch.randn((2, 3, 5, 5), dtype=torch.float32, device=torch.device("cpu")).to(memory_format=torch.channels_last)
        x_gpu = x_cpu.to("xpu").to(memory_format=torch.channels_last)
        scales = [6, 8]
        rsf = False

        dtype_inputs = torch.qint8
        q_scale = 0.04
        q_cpu = torch.quantize_per_tensor(x_cpu, q_scale, 0, dtype_inputs).to(memory_format=torch.channels_last)
        q_gpu = torch.quantize_per_tensor(x_gpu, q_scale, 0, dtype_inputs).to(memory_format=torch.channels_last)

        output_cpu = torch.nn.functional.interpolate(
            q_cpu, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)
        output_gpu = torch.nn.functional.interpolate(
            q_gpu, scale_factor=scales, mode='nearest', recompute_scale_factor=rsf)

        self.assertEqual(output_cpu, output_gpu)
