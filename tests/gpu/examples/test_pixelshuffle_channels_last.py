import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch.testing._internal.common_utils import TestCase

import ipex

import pytest


class PixelShuffle(torch.nn.Module):
    def __init__(self):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel = nn.PixelShuffle(2)

    def forward(self, x):
        res = self.conv(x)
        res = self.pixel(res)
        return res

class TestTorchMethod(TestCase):
    @pytest.mark.skipif(torch.xpu.using_onednn_layout(), reason="channels last does not support onednn block format")
    def test_dequant_pixelshuffle(self, dtype=torch.float):
        src_cpu = torch.randn(1, 64, 64, 64)
        src_xpu = src_cpu.to(memory_format=torch.channels_last).to("xpu")
        model = PixelShuffle()
        dst_cpu = model(src_cpu)
        model_xpu = model.to("xpu")
        dst_xpu = model_xpu(src_xpu)

        self.assertEqual(dst_xpu.is_contiguous(memory_format=torch.channels_last), True)
        self.assertEqual(dst_cpu, dst_xpu.cpu())
