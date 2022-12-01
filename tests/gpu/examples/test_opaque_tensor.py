import torch
import torch.nn as nn
import pytest
import intel_extension_for_pytorch # noqa
from torch.testing._internal.common_utils import TestCase

class TestTorchMethod(TestCase):
    def test_activation(self, dtype=torch.float):
        conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False).to("xpu")
        conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False).to("xpu")
        x = torch.randn(16, 64, 56, 56, dtype=dtype, device=torch.device("xpu"))

        ref = conv2(conv1(x))

        torch.xpu.utils.enable_onednn_layout()
        with torch.inference_mode():
            y = conv2(conv1(x))
        torch.xpu.utils.disable_onednn_layout()

        self.assertEqual(y.cpu(), ref.cpu())


    def test_weight_channels_last(self, dtype=torch.float):
        conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ref = conv1.weight.detach().clone().contiguous(memory_format=torch.channels_last)
        conv1.weight.data = conv1.weight.data.contiguous(memory_format=torch.channels_last).to("xpu")
        x = torch.randn(16, 64, 56, 56, dtype=dtype, device=torch.device("xpu"))

        torch.xpu.utils.enable_onednn_layout()
        with torch.inference_mode():
            y = conv1(x)
            y = conv1(x)
        torch.xpu.utils.disable_onednn_layout()

        w = conv1.weight * 2.22 # to_plain to check meta
        ref = ref * 2.22

        self.assertEqual(w.shape, ref.shape)
        self.assertEqual(w.stride(), ref.stride())
        self.assertEqual(w.cpu(), ref.cpu())


    def test_group_conv(self, dtype=torch.float):
        conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=2, bias=False).to("xpu")
        ref = conv1.weight.detach().clone()
        x = torch.randn(16, 64, 56, 56, dtype=dtype, device=torch.device("xpu"))

        torch.xpu.utils.enable_onednn_layout()
        with torch.inference_mode():
            y = conv1(x)
            y = conv1(x)
        torch.xpu.utils.disable_onednn_layout()

        w = conv1.weight * 2.22 # to_plain to check meta
        ref = ref * 2.22

        self.assertEqual(w.shape, ref.shape)
        self.assertEqual(w.stride(), ref.stride())
        self.assertEqual(w.cpu(), ref.cpu())


    def test_linear_relu(self, dtype=torch.int8):
        a = torch.randn(2048, 2048)
        b = torch.randn(2048, 2048)
        a = a / a.max() * 127
        b = b / b.max() * 127
        a = a.to(torch.int8).to("xpu")
        b = b.to(torch.int8).to("xpu")

        ref = b.detach().clone()

        torch.xpu.utils.enable_onednn_layout()
        with torch.inference_mode():
            c = torch.ops.torch_ipex.linear_relu(a, b, a)
            c = torch.ops.torch_ipex.linear_relu(a, b, a)
        torch.xpu.utils.disable_onednn_layout()

        b = b * 2.22 # to_plain to check meta
        ref = ref * 2.22

        self.assertEqual(b.shape, ref.shape)
        self.assertEqual(b.stride(), ref.stride())
        self.assertEqual(b.cpu(), ref.cpu())
