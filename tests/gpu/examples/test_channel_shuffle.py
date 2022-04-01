import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import pytest


class ChannelShuffle(torch.nn.Module):
    def __init__(self):
        super(ChannelShuffle, self).__init__()
        self.channelShuffle = nn.ChannelShuffle(2)

    def forward(self, x):
        res = self.channelShuffle(x)
        return res

class TestTorchMethod(TestCase):
    def test_channelshuffle(self, dtype=torch.float):
        src_cpu = torch.randn(1, 4, 2, 2)
        src_xpu = src_cpu.to("xpu")
        model = ChannelShuffle()
        dst_cpu = model(src_cpu)
        model_xpu = model.to("xpu")
        dst_xpu = model_xpu(src_xpu)

        self.assertEqual(dst_xpu.is_contiguous(), True)
        self.assertEqual(dst_cpu, dst_xpu.cpu())

    @pytest.mark.skipif(torch.xpu.using_layout_opt(), reason="channels last does not support onednn block format")
    def test_channelshuffle_channels_last(self, dtype=torch.float):
        src_cpu = torch.randn(1, 4, 2, 2)
        src_xpu = src_cpu.to(memory_format=torch.channels_last).to("xpu")
        model = ChannelShuffle()
        dst_cpu = model(src_cpu)
        model_xpu = model.to("xpu")
        dst_xpu = model_xpu(src_xpu)

        self.assertEqual(dst_xpu.is_contiguous(memory_format=torch.channels_last), True)
        self.assertEqual(dst_cpu, dst_xpu.cpu())
