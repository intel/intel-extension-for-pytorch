import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch as ipex  # noqa
from intel_extension_for_pytorch.quantization.fp8.util import (
    cast_to_fp8,
)

import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skip(
        reason="PT2.5: 'index_select_xpu' not implemented for 'Float8_e4m3fn'"
    )
    def test_index_select(self, dtype=torch.float):
        dim_size = 10
        dims = 3

        def _test_index_select(input, input_fp8, indcies):
            def _test(input, indcies, dim):
                y_ref = torch.index_select(input, dim, indices)
                y_ref = cast_to_fp8(y_ref, fp8_meta["test"], fp8_tensor, fp8_dtype)
                y = torch.index_select(input_fp8, dim, indcies)
                # print("y_ref = ", y_ref)
                # print("y = ", y)
                self.assertEqual(y_ref.cpu(), y.cpu())

            _test(input, indcies, 0)
            _test(input, indcies, 1)

        a = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5],
            ],
            dtype=dtype,
        ).xpu()
        indices = torch.LongTensor([0, 2]).xpu()

        fp8_dtype = ipex._isa_help.Float8Format.kFloat8_E4M3

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

        # print("a = ", a)
        a_quantized = cast_to_fp8(a, fp8_meta["test"], fp8_tensor, fp8_dtype)
        # print("a_quantized = ", a_quantized)
        # print("max = ", fp8_meta["test"].amax_history)
        # print("scale = ", fp8_meta["test"].scale)
        # print("scale_iv = ", fp8_meta["test"].scale_inv)

        _test_index_select(a, a_quantized, indices)
