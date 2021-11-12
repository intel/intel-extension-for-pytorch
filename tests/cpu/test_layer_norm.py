import unittest

import torch
import intel_extension_for_pytorch as ipex
from common_utils import TestCase


class M1(torch.nn.Module):
    def __init__(self):
        super(M1, self).__init__()
        self.conv = torch.nn.Conv2d(5, 5, 1, stride=1, bias=False)
        self.layer_norm = torch.nn.LayerNorm(10)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer_norm(x)
        return x

class M2(torch.nn.Module):
    def __init__(self):
        super(M2, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(10)

    def forward(self, x):
        x = self.layer_norm(x)
        return x

class LayerNormTester(TestCase):
    def test_layer_norm(self):
        # autocast inference path. layer_norm is fallthrough.
        with torch.cpu.amp.autocast(), torch.no_grad():
            x = torch.randn(20, 5, 10, 10)
            # layernorm input is bfloat16
            # layernomr is in blacklist, so output is fp32
            model = M1().eval()
            trace_model = torch.jit.trace(model, x)
            y1_bf16 = model(x)
            y2_bf16 = trace_model(x)
            self.assertEqual(y1_bf16.dtype, torch.float32)
            self.assertEqual(y2_bf16.dtype, torch.float32)
            self.assertEqual(y1_bf16, y2_bf16, prec=0.03)

            # layernorm input is fp32
            model = M2().eval()
            trace_model = torch.jit.trace(model, x)
            y1_fp32 = model(x)
            y2_fp32 = trace_model(x)
            self.assertEqual(y1_fp32.dtype, torch.float32)
            self.assertEqual(y2_fp32.dtype, torch.float32)
            self.assertEqual(y1_fp32, y2_fp32)

if __name__ == '__main__':
    test = unittest.main()
