import torch
import intel_extension_for_pytorch  # noqa
from enum import Enum

from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


class QuantMode(Enum):
    SYM = 1
    ASYM = 2


checking_atol = 1e-1
checking_rtol = 1e-1


class TestDynamicQuant(TestCase):

    def dynamic_per_token_quant_ref(self, input, use_sym_quant, bits):
        k = input.shape[-1]
        qmin = -(2 ** (bits - 1)) if use_sym_quant else 0
        qmax = 2 ** (bits - 1) - 1 if use_sym_quant else 2**bits - 1
        min_val = torch.min(input, dim=-1)[0].to(dtype=torch.float32)
        max_val = torch.max(input, dim=-1)[0].to(dtype=torch.float32)
        if use_sym_quant:
            scale = torch.maximum(torch.abs(min_val), torch.abs(max_val)) / qmax
            zero_point = torch.zeros_like(scale).to(dtype=torch.int32)
        else:
            scale = (max_val - min_val) / qmax
            zero_point = -1 * torch.round(min_val / scale).to(dtype=torch.int32)
        scale = scale.to(dtype=input.dtype)
        quantized = torch.clamp(
            torch.round(
                input / scale.unsqueeze(-1).repeat(1, k).to(dtype=torch.float32)
                + zero_point.unsqueeze(-1).repeat(1, k)
            ),
            qmin,
            qmax,
        ).to(dtype=torch.int8 if use_sym_quant else torch.uint8)
        return quantized, scale, zero_point

    def dequant(self, quantized, scale, zero_point):
        k = quantized.shape[-1]
        return (quantized - zero_point.repeat(1, k)) * scale.repeat(1, k)

    @parametrize(
        "m, k",
        [(3, 4), (5, 10), (8, 8), (64, 128), (64, 255), (1024, 4096), (1024, 14336)],
    )
    @parametrize("dtype", [torch.float32, torch.float16])
    @parametrize("quant_mode", [QuantMode.SYM, QuantMode.ASYM])
    def test_dynamic_quant(self, m, k, dtype, quant_mode):
        torch.manual_seed(0)
        input = torch.randn([m, k], dtype=dtype, device="xpu")
        use_sym_quant = quant_mode == QuantMode.SYM
        bits = 8  # Assuming 8-bit quantization

        quantized, scale, zero_point = torch.ops.torch_ipex.dynamic_per_token_quant(
            input, use_sym_quant
        )
        dequantized = self.dequant(quantized, scale, zero_point)
        self.assertEqual(dequantized, input, atol=checking_atol, rtol=checking_rtol)


instantiate_parametrized_tests(TestDynamicQuant)

if __name__ == "__main__":
    run_tests()
