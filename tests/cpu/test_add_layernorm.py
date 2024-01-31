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
                for dtype in [torch.bfloat16, torch.float16]:
                    with torch.cpu.amp.autocast(dtype=dtype), torch.no_grad():
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

                        # add_layernorm input is bfloat16/float16
                        a_lowp = a.to(dtype=dtype)
                        b_lowp = b.to(dtype=dtype)
                        model = model.to(dtype=dtype)
                        trace_model = torch.jit.trace(model, (a_lowp, b_lowp))
                        y1_lowp = model(a_lowp, b_lowp)
                        y2_lowp = trace_model(a_lowp, b_lowp)
                        self.assertEqual(y1_lowp.dtype, dtype)
                        self.assertEqual(y2_lowp.dtype, dtype)
                        # Add a custom threshold for bf16/fp16 test because of fused add_layernorm in jit has higher precision
                        # and causes mismatch with eager mode.
                        prec = 5e-2 if dtype == torch.bfloat16 else 5e-3
                        self.assertEqual(y1_lowp, y2_lowp, prec=prec)


if __name__ == "__main__":
    test = unittest.main()
