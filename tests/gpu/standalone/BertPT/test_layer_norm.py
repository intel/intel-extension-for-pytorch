import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import copy

dpcpp_device = torch.device("xpu")
cpu_device = torch.device("cpu")

class TestTorchMethod(TestCase):
    def test_layer_norm(self, dtype=torch.float):
        layer_norm = nn.LayerNorm(1024)
        x = torch.randn([1, 512, 1024], device=cpu_device, dtype=torch.float)
        x_xpu = x.clone().xpu()
        y = layer_norm(x)
        layer_norm_xpu = nn.LayerNorm(1024).xpu()
        y_xpu = layer_norm_xpu(x_xpu)
        self.assertEqual(y, y_xpu.cpu())

