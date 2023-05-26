import unittest

import torch
from common_utils import TestCase


class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(10)

    def forward(self, x):
        x = self.layer_norm(x)
        return x


class LayerNormTester(TestCase):
    def test_layer_norm(self):
        # autocast inference path. layer_norm is fallthrough.
        for dim in [2, 3, 4, 5, 6, 7]:
            for full_bf16 in [False, True]:
                model = M().eval()
                if full_bf16:  # support full bf16 mode for layer_norm
                    model = model.bfloat16()
                with torch.cpu.amp.autocast(), torch.no_grad():
                    input_size = [
                        3,
                    ]
                    for _ in range(dim - 1):
                        input_size.append(10)
                    x = torch.randn(input_size)
                    x_bf16 = x.bfloat16()
                    # layernorm input is bfloat16
                    trace_model = torch.jit.trace(model, x_bf16)
                    y1_bf16 = model(x_bf16)
                    y2_bf16 = trace_model(x_bf16)
                    self.assertEqual(y1_bf16.dtype, torch.bfloat16)
                    self.assertEqual(y2_bf16.dtype, torch.bfloat16)
                    self.assertEqual(y1_bf16, y2_bf16)
                    if not full_bf16:
                        # layernorm input is fp32
                        trace_model = torch.jit.trace(model, x)
                        y1_fp32 = model(x)
                        y2_fp32 = trace_model(x)
                        self.assertEqual(y1_fp32.dtype, torch.float32)
                        self.assertEqual(y2_fp32.dtype, torch.float32)
                        self.assertEqual(y1_fp32, y2_fp32)


if __name__ == "__main__":
    test = unittest.main()
