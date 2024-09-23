import torch
import torch.nn as nn
from common_utils import TestCase
import unittest
import itertools


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=torch.float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(
        self, hidden_states, fused_rmsnorm=False, extra_input=None, add_back=False
    ):
        if fused_rmsnorm:
            if extra_input is None:
                return torch.ops.torch_ipex.rmsnorm(
                    hidden_states, self.weight, self.variance_epsilon
                )
            else:
                return torch.ops.torch_ipex.add_rmsnorm(
                    hidden_states,
                    extra_input,
                    self.weight,
                    self.variance_epsilon,
                    add_back,
                )
        else:
            if extra_input is not None:
                if add_back:
                    extra_input.add_(hidden_states)
                    hidden_states = extra_input
                else:
                    hidden_states = hidden_states + extra_input
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            res = (self.weight * hidden_states).to(input_dtype)
            return res


class RMSNormTester(TestCase):
    def test_RMSNorm(self):
        for dim in [2, 3, 4, 5]:
            # RMSNorm input is fp32
            for weight_dtype in [torch.float32, torch.half, torch.bfloat16]:
                with torch.no_grad():
                    input_size = [
                        3,
                    ]
                    for _ in range(dim - 1):
                        input_size.append(10)
                    x = torch.randn(input_size)
                    model = RMSNorm(input_size, dtype=weight_dtype).eval()
                    y1_fp32 = model(x)
                    fused_y1_fp32 = model(x, fused_rmsnorm=True)
                    self.assertEqual(y1_fp32, fused_y1_fp32)
            # RMSNorm input is bf16
            for weight_dtype in [torch.float32, torch.half, torch.bfloat16]:
                with torch.no_grad():
                    input_size = [
                        3,
                    ]
                    for _ in range(dim - 1):
                        input_size.append(10)
                    x = torch.randn(input_size)

                    model = RMSNorm(input_size, dtype=weight_dtype).eval()
                    x_bf16 = x.to(torch.bfloat16)
                    y1_bf16 = model(x_bf16)
                    fused_y1_bf16 = model(x_bf16, fused_rmsnorm=True)
                    self.assertEqual(y1_bf16, fused_y1_bf16, prec=1e-2)
            # RMSNorm input is fp16
            for weight_dtype in [torch.float32, torch.half, torch.bfloat16]:
                with torch.no_grad():
                    input_size = [
                        3,
                    ]
                    for _ in range(dim - 1):
                        input_size.append(10)
                    x = torch.randn(input_size)
                    model = RMSNorm(input_size, dtype=weight_dtype).eval()
                    x_fp16 = x.to(torch.half)
                    y1_fp16 = model(x_fp16)
                    fused_y1_fp16 = model(x_fp16, fused_rmsnorm=True)
                    self.assertEqual(y1_fp16, fused_y1_fp16, prec=1e-2)

    def test_add_RMSNorm(self):
        add_back_list = [False, True]
        dim_list = [2, 3, 4, 5]
        cases = itertools.product(add_back_list, dim_list)
        for add_back, dim in cases:
            # RMSNorm input is fp32
            for weight_dtype in [torch.float32, torch.half, torch.bfloat16]:
                with torch.no_grad():
                    input_size = [
                        3,
                    ]
                    for _ in range(dim - 1):
                        input_size.append(20)
                    x = torch.randn(input_size)
                    x1 = torch.randn(input_size)
                    model = RMSNorm(input_size, dtype=weight_dtype).eval()
                    x2 = x1.clone()
                    y1_fp32 = model(x, extra_input=x1, add_back=add_back)
                    fused_y1_fp32 = model(
                        x, fused_rmsnorm=True, extra_input=x2, add_back=add_back
                    )
                    self.assertEqual(y1_fp32, fused_y1_fp32)
                    self.assertEqual(x1, x2)
            # RMSNorm input is bf16
            for weight_dtype in [torch.float32, torch.half, torch.bfloat16]:
                with torch.no_grad():
                    input_size = [
                        3,
                    ]
                    for _ in range(dim - 1):
                        input_size.append(20)
                    x_bf16 = torch.randn(input_size, dtype=torch.bfloat16)
                    x1_bf16 = torch.randn(input_size, dtype=torch.bfloat16)
                    model = RMSNorm(input_size, dtype=weight_dtype).eval()
                    x2_bf16 = x1_bf16.clone()
                    y1_bf16 = model(x_bf16, extra_input=x1_bf16, add_back=add_back)
                    fused_y1_bf16 = model(
                        x_bf16,
                        fused_rmsnorm=True,
                        extra_input=x2_bf16,
                        add_back=add_back,
                    )
                    self.assertEqual(y1_bf16, fused_y1_bf16, prec=2e-2)
                    self.assertEqual(x1_bf16, x2_bf16)
            # RMSNorm input is fp16
            for weight_dtype in [torch.float32, torch.half, torch.bfloat16]:
                with torch.no_grad():
                    input_size = [
                        3,
                    ]
                    for _ in range(dim - 1):
                        input_size.append(20)
                    x_fp16 = torch.randn(input_size, dtype=torch.half)
                    x1_fp16 = torch.randn(input_size, dtype=torch.half)
                    model = RMSNorm(input_size, dtype=weight_dtype).eval()
                    x2_fp16 = x1_fp16.clone()
                    y1_fp16 = model(x_fp16, extra_input=x1_fp16, add_back=add_back)
                    fused_y1_fp16 = model(
                        x_fp16,
                        fused_rmsnorm=True,
                        extra_input=x2_fp16,
                        add_back=add_back,
                    )
                    self.assertEqual(y1_fp16, fused_y1_fp16, prec=1e-2)
                    self.assertEqual(x1_fp16, x2_fp16)


if __name__ == "__main__":
    test = unittest.main()
