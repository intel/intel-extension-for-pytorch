import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import pytest
import ipex
import tempfile


class Conv2dRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return F.relu(self.conv(x) + a, inplace=True)


class TestJitSaveLoadMethod(TestCase):
    def test_jit_save_load(self):
        # // you can find a simple model in tests/example/test_fusion.py
        model = Conv2dRelu(2, 2, kernel_size=3, stride=1, bias=True)
        model = model.to('xpu').eval()
        origin_modelJit = torch.jit.script(model)
        ckpt = tempfile.NamedTemporaryFile()
        origin_modelJit.save(ckpt.name)
        loaded_modelJit = torch.jit.load(ckpt.name)

        for o, l in zip(origin_modelJit.parameters(), loaded_modelJit.parameters()):
            print(f"o: {o.cpu()}")
            print(f"l: {l.cpu()}")
            assert torch.equal(o, l), " param tensor in saved & loaded not equal"
