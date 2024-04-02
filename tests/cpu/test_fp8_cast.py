import torch
import unittest
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization.fp8.util import cast_to_fp8, cast_from_fp8
from torch.testing._internal.common_utils import TestCase


class TestFP8Cast(TestCase):
    def test_fp8_cast(self):
        out_dtype = torch.float
        for fp8_dtype in {
            ipex._isa_help.Float8Format.kFloat8_E4M3,
            ipex._isa_help.Float8Format.kFloat8_E5M2,
        }:
            a = torch.ones([2, 5])
            fp8_meta = {}
            fp8_meta["test"] = ipex._isa_help.FP8TensorMeta()
            fp8_meta["test"].scale = torch.ones(1) * 2
            fp8_meta["test"].scale_inv = torch.ones(1)
            fp8_meta["test"].amax_history = torch.ones([1, 4])

            fp8_tensor = ipex._isa_help.FP8FwdTensors.GEMM1_INPUT

            print("a = ", a)
            a_quantized = cast_to_fp8(a, fp8_meta["test"], fp8_tensor, fp8_dtype)
            print("a_quantized = ", a_quantized)
            print("max = ", fp8_meta["test"].amax_history)
            print("scale = ", fp8_meta["test"].scale)
            print("scale_inv = ", fp8_meta["test"].scale_inv)
            a_dequantized = cast_from_fp8(
                a_quantized, fp8_meta["test"], fp8_tensor, fp8_dtype, torch.float
            )
            print("a_dequtized = ", a_dequantized)
            self.assertEqual(a, a_dequantized)


if __name__ == "__main__":
    test = unittest.main()
