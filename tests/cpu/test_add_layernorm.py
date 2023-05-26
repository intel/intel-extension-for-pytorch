import unittest

import torch
from common_utils import TestCase


class add_layernorm(torch.nn.Module):
    def __init__(self, size):
        super(add_layernorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(size)

    def forward(self, a, b):
        x = torch.add(a, b)
        x = self.layer_norm(x)
        return x


class AddLayerNormTester(TestCase):
    def test_add_layernorm(self):
        for size in [10, 16, 35]:
            for dim in [2, 3, 4, 5]:
                with torch.cpu.amp.autocast(), torch.no_grad():
                    input_size = [
                        3,
                    ]
                    for _ in range(dim - 1):
                        input_size.append(size)
                    # add_layernorm input is fp32
                    a = torch.randn(input_size)
                    b = torch.randn(input_size)
                    model = add_layernorm(size).eval()
                    trace_model = torch.jit.trace(model, (a, b))
                    y1_fp32 = model(a, b)
                    y2_fp32 = trace_model(a, b)
                    self.assertEqual(y1_fp32.dtype, torch.float32)
                    self.assertEqual(y2_fp32.dtype, torch.float32)
                    self.assertEqual(y1_fp32, y2_fp32)

                    # add_layernorm input is bfloat16
                    a_bf16 = a.bfloat16()
                    b_bf16 = b.bfloat16()
                    model = model.bfloat16()
                    trace_model = torch.jit.trace(model, (a_bf16, b_bf16))
                    y1_bf16 = model(a_bf16, b_bf16)
                    y2_bf16 = trace_model(a_bf16, b_bf16)
                    self.assertEqual(y1_bf16.dtype, torch.bfloat16)
                    self.assertEqual(y2_bf16.dtype, torch.bfloat16)
                    # Add a custom threshold for bf16 test because of fused add_layernorm in jit has higher precision
                    # and causes mismatch with eager mode.
                    self.assertEqual(y1_bf16, y2_bf16, prec=5e-2)


if __name__ == "__main__":
    test = unittest.main()
