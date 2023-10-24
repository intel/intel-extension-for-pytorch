import torch
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.xpu.fp8.utils import cast_to_fp8, cast_from_fp8
from torch.testing._internal.common_utils import TestCase


class TestFP8Cast(TestCase):
    def test_fp8_e4m3_cast(self):
        out_dtype = torch.float
        fp8_dtype = ipex._isa_help.Float8Format.kFloat8_E4M3

        a = torch.ones([2, 5]).xpu()
        a_scale, a_amax, a_scale_inv = (
            torch.ones([]).xpu(),
            torch.zeros([]).xpu(),
            torch.ones([]).xpu(),
        )

        fp8_meta = {}
        fp8_meta["test"] = ipex._isa_help.FP8TensorMeta()
        fp8_meta["test"].scale = (torch.ones(1) * 2).xpu()
        fp8_meta["test"].scale_inv = torch.ones(1).xpu()
        fp8_meta["test"].amax_history = torch.ones([1, 4]).xpu()

        fp8_tensor = ipex._isa_help.FP8FwdTensors.GEMM1_INPUT

        print("a = ", a)
        a_quantized = cast_to_fp8(a, fp8_meta["test"], fp8_tensor, fp8_dtype)
        print("a_quantized = ", a_quantized)
        print("max = ", fp8_meta["test"].amax_history)
        print("scale = ", fp8_meta["test"].scale)
        print("scale_iv = ", fp8_meta["test"].scale_inv)

        a_dequtized = cast_from_fp8(
            a_quantized, fp8_meta["test"], fp8_tensor, fp8_dtype, torch.float
        )
        print("a_dequtized = ", a_dequtized)

        self.assertEqual(a, a_dequtized)

    def test_fp8_e5m2_cast(self):
        out_dtype = torch.float
        fp8_dtype = ipex._isa_help.Float8Format.kFloat8_E5M2

        a = torch.ones([2, 5]).xpu()
        a_scale, a_amax, a_scale_inv = (
            torch.ones([]).xpu(),
            torch.zeros([]).xpu(),
            torch.ones([]).xpu(),
        )

        fp8_meta = {}
        fp8_meta["test"] = ipex._isa_help.FP8TensorMeta()
        fp8_meta["test"].scale = (torch.ones(1) * 2).xpu()
        fp8_meta["test"].scale_inv = torch.ones(1).xpu()
        fp8_meta["test"].amax_history = torch.ones([1, 4]).xpu()

        fp8_tensor = ipex._isa_help.FP8FwdTensors.GEMM1_INPUT

        print("a = ", a)
        a_quantized = cast_to_fp8(a, fp8_meta["test"], fp8_tensor, fp8_dtype)
        print("a_quantized = ", a_quantized)
        print("max = ", fp8_meta["test"].amax_history)
        print("scale = ", fp8_meta["test"].scale)
        print("scale_iv = ", fp8_meta["test"].scale_inv)

        a_dequtized = cast_from_fp8(
            a_quantized, fp8_meta["test"], fp8_tensor, fp8_dtype, torch.float
        )
        print("a_dequtized = ", a_dequtized)

        self.assertEqual(a, a_dequtized)
