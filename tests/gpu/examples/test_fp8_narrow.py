import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch as ipex  # noqa
from intel_extension_for_pytorch.xpu.fp8.utils import (
    cast_to_fp8,
)


class TestTorchMethod(TestCase):
    def test_fp8_narrow(self, dtype=torch.float):
        fp8_dtype = ipex._isa_help.Float8Format.kFloat8_E4M3
        fp8_meta = {}
        fp8_meta["test"] = ipex._isa_help.FP8TensorMeta()
        fp8_meta["test"].scale = (torch.ones(1) * 2).xpu()
        fp8_meta["test"].scale_inv = torch.ones(1).xpu()
        fp8_meta["test"].amax_history = torch.ones([1, 4]).xpu()

        fp8_tensor = ipex._isa_help.FP8FwdTensors.GEMM1_INPUT

        a = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5],
            ],
            dtype=torch.float,
        ).xpu()
        # print("a = ", a)
        a_quantized = cast_to_fp8(a, fp8_meta["test"], fp8_tensor, fp8_dtype)
        # print("a_quantized = ", a_quantized)

        b_ref = a.narrow(1, 1, 3)
        # print("b_ref = ", b_ref)
        b_ref = cast_to_fp8(b_ref, fp8_meta["test"], fp8_tensor, fp8_dtype)
        # print("b_ref = ", b_ref)

        b = a_quantized.narrow(1, 1, 3)

        # print("b = ", b)
        self.assertEqual(b.cpu(), b_ref.cpu())
