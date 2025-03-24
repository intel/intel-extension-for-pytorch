################################################################################
# Copyright (C) 2024 Intel Corporation
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written
# permission. This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly
# stated in the License.
################################################################################
import torch
import intel_extension_for_pytorch as ipex  # noqa
import pytest

# from intel_extension_for_pytorch.xpu.fp8.util import cast_to_fp8, cast_from_fp8, cast_to_fp8_hybrid
from torch.testing._internal.common_utils import TestCase


class TestFP8CastV2(TestCase):
    @pytest.mark.skipif(True, reason="SR depend on ESIMD, skip before sycl support")
    def test_fp8_e4m3_cast_v2_float(self, dtype=torch.float):
        fp8_dtype = torch.float8_e5m2

        a = torch.randint(1, 5, [2, 5]).xpu().to(dtype)
        scale = (torch.ones(1) * 2).xpu()
        scale_inv = torch.tensor([1 / 2]).xpu()
        scale_shape = None

        # print("a = ", a)
        amax_ref = a.abs().max().float()
        # print("max_ref = ", amax_ref)
        a_quantized, amax = torch.ops.torch_ipex.cast_to_fp8(
            a, scale, True, True, fp8_dtype, scale_shape
        )
        # print("a_quantized = ", a_quantized)
        # print("scale = ", scale)
        # print("scale_inv = ", scale_inv)

        a_dequtized = torch.ops.torch_ipex.cast_from_fp8(a_quantized, scale_inv, dtype)
        # print("a_dequtized = ", a_dequtized)

        self.assertEqual(a, a_dequtized.cpu())
        self.assertEqual(amax, amax_ref)

    def test_fp8_e5m2_cast_v2_float(self, dtype=torch.float):
        fp8_dtype = torch.float8_e5m2

        a = torch.randint(1, 5, [2, 5]).xpu().to(dtype)
        scale = None
        scale_inv = None
        scale_shape = None

        # print("a = ", a)
        a_quantized, amax = torch.ops.torch_ipex.cast_to_fp8(
            a, scale, False, False, fp8_dtype, scale_shape
        )
        # print("a_quantized = ", a_quantized)
        # print("scale = ", scale)
        # print("scale_inv = ", scale_inv)

        a_dequtized = torch.ops.torch_ipex.cast_from_fp8(a_quantized, scale_inv, dtype)
        # print("a_dequtized = ", a_dequtized)

        self.assertEqual(a, a_dequtized.cpu())

    @pytest.mark.skipif(True, reason="SR depend on ESIMD, skip before sycl support")
    def test_fp8_e4m3_cast_v2_bfloat16(self, dtype=torch.bfloat16):
        fp8_dtype = torch.float8_e4m3fn

        a = torch.randint(1, 5, [2, 5]).xpu().to(dtype)
        scale = (torch.ones(1) * 2).xpu()
        scale_inv = torch.tensor([1 / 2]).xpu()
        scale_shape = None

        # print("a = ", a)
        a_quantized, amax = torch.ops.torch_ipex.cast_to_fp8(
            a, scale, True, False, fp8_dtype, scale_shape
        )
        # print("a_quantized = ", a_quantized)
        # print("scale = ", scale)
        # print("scale_inv = ", scale_inv)

        a_dequtized = torch.ops.torch_ipex.cast_from_fp8(a_quantized, scale_inv, dtype)
        # print("a_dequtized = ", a_dequtized)

        self.assertEqual(a, a_dequtized.cpu())

    @pytest.mark.skipif(True, reason="SR depend on ESIMD, skip before sycl support")
    def test_fp8_e5m2_cast_v2_bfloat16(self, dtype=torch.bfloat16):
        fp8_dtype = torch.float8_e5m2

        a = torch.randint(1, 5, [2, 5]).xpu().to(dtype)
        scale = (torch.ones(1) * 2).xpu().bfloat16()
        scale_inv = torch.tensor([1 / 2]).xpu().bfloat16()
        scale_shape = None

        # print("a = ", a)
        amax_ref = a.abs().max().float()
        # print("max_ref = ", amax_ref)
        a_quantized, amax = torch.ops.torch_ipex.cast_to_fp8(
            a, scale, True, True, fp8_dtype, scale_shape
        )
        # print("a_quantized = ", a_quantized)
        # print("scale = ", scale)
        # print("scale_inv = ", scale_inv)

        a_dequtized = torch.ops.torch_ipex.cast_from_fp8(a_quantized, scale_inv, dtype)
        # print("a_dequtized = ", a_dequtized)

        self.assertEqual(a, a_dequtized.cpu())
        self.assertEqual(amax, amax_ref)

    def test_fp8_e4m3_cast_v2_float16(self, dtype=torch.float16):
        fp8_dtype = torch.float8_e5m2

        a = torch.randint(1, 5, [2, 5]).xpu().to(dtype)
        scale = (torch.ones(1) * 2).xpu()
        scale_inv = torch.tensor([1 / 2]).xpu().bfloat16()
        scale_shape = None

        # print("a = ", a)
        a_quantized, amax = torch.ops.torch_ipex.cast_to_fp8(
            a, scale, False, False, fp8_dtype, scale_shape
        )
        # print("a_quantized = ", a_quantized)
        # print("scale = ", scale)
        # print("scale_inv = ", scale_inv)

        a_dequtized = torch.ops.torch_ipex.cast_from_fp8(a_quantized, scale_inv, dtype)
        # print("a_dequtized = ", a_dequtized)

        self.assertEqual(a, a_dequtized.cpu())

    @pytest.mark.skipif(True, reason="SR depend on ESIMD, skip before sycl support")
    def test_fp8_e5m2_cast_v2_float16(self, dtype=torch.float16):
        fp8_dtype = torch.float8_e5m2

        a = torch.randint(1, 5, [2, 5]).xpu().to(dtype)
        scale = (torch.ones(1) * 2).xpu().half()
        scale_inv = torch.tensor([1 / 2]).xpu()
        scale_shape = None

        # print("a = ", a)
        a_quantized, amax = torch.ops.torch_ipex.cast_to_fp8(
            a, scale, True, False, fp8_dtype, scale_shape
        )
        # print("a_quantized = ", a_quantized)
        # print("scale = ", scale)
        # print("scale_inv = ", scale_inv)

        a_dequtized = torch.ops.torch_ipex.cast_from_fp8(a_quantized, scale_inv, dtype)
        # print("a_dequtized = ", a_dequtized)

        self.assertEqual(a, a_dequtized.cpu())

    @pytest.mark.skipif(True, reason="SR depend on ESIMD, skip before sycl support")
    def test_fp8_e4m3_cast_v2_scalar(self, dtype=torch.float16):
        fp8_dtype = torch.float8_e4m3fn

        a = torch.randint(1, 5, [2, 5]).xpu().to(dtype)
        scale = 2
        scale_inv = 0.5
        scale_shape = None

        # print("a = ", a)
        amax_ref = a.abs().max().float()
        # print("max_ref = ", amax_ref)
        a_quantized, amax = torch.ops.torch_ipex.cast_to_fp8(
            a, scale, True, True, fp8_dtype, scale_shape
        )
        # print("a_quantized = ", a_quantized)
        # print("amax = ", amax)
        # print("scale = ", scale)
        # print("scale_inv = ", scale_inv)

        a_dequtized = torch.ops.torch_ipex.cast_from_fp8(a_quantized, scale_inv, dtype)
        # print("a_dequtized = ", a_dequtized)

        self.assertEqual(a, a_dequtized)
        self.assertEqual(amax, amax_ref)

    def test_fp8_e5m2_cast_v2_scalar(self, dtype=torch.bfloat16):
        fp8_dtype = torch.float8_e4m3fn

        a = torch.randint(1, 5, [2, 5]).xpu().to(dtype)
        scale = None
        scale_inv = None
        scale_shape = None

        # print("a = ", a)
        amax_ref = a.abs().max().float()
        # print("max_ref = ", amax_ref)
        a_quantized, amax = torch.ops.torch_ipex.cast_to_fp8(
            a, scale, False, True, fp8_dtype, scale_shape
        )
        # print("a_quantized = ", a_quantized)
        # print("amax = ", amax)
        # print("scale = ", scale)
        # print("scale_inv = ", scale_inv)

        a_dequtized = torch.ops.torch_ipex.cast_from_fp8(a_quantized, scale_inv, dtype)
        # print("a_dequtized = ", a_dequtized)

        self.assertEqual(a, a_dequtized)
        self.assertEqual(amax, amax_ref)

    @pytest.mark.skipif(True, reason="SR depend on ESIMD, skip before sycl support")
    def test_fp8_cast_hybrid(self, dtype=torch.float):
        input = torch.randint(1, 5, [2, 5]).xpu().to(dtype)
        scale_152 = (torch.ones(1) * 4).xpu()
        scale_inv_152 = torch.tensor([1 / 4]).xpu()

        scale_143 = (torch.ones(1) * 2).xpu()
        scale_inv_143 = torch.tensor([1 / 2]).xpu()
        # print("input = ", input)
        amax_ref = input.abs().max().float()
        # print("max_ref = ", amax_ref)
        quantized_152, quantized_143, amax = torch.ops.torch_ipex.cast_to_fp8_hybrid(
            input, scale_152, scale_143, True, True
        )
        # print("quantized_152 = ", quantized_152)
        # print("quantized_143 = ", quantized_143)
        # print("amax = ", amax)

        dequantized_152 = torch.ops.torch_ipex.cast_from_fp8(
            quantized_152, scale_inv_152, dtype
        )
        dequantized_143 = torch.ops.torch_ipex.cast_from_fp8(
            quantized_143, scale_inv_143, dtype
        )
        # print("dequantized_152 = ", dequantized_152)
        # print("dequantized_143 = ", dequantized_143)

        self.assertEqual(input, dequantized_152)
        self.assertEqual(input, dequantized_143)
        self.assertEqual(amax, amax_ref)

    def test_fp8_cast_hybrid_none(self, dtype=torch.bfloat16):
        input = torch.randint(1, 5, [2, 5]).xpu().to(dtype)
        scale_152 = None
        scale_inv_152 = None

        scale_143 = None
        scale_inv_143 = None
        # print("input = ", input)
        quantized_152, quantized_143, amax = torch.ops.torch_ipex.cast_to_fp8_hybrid(
            input, scale_152, scale_143, False, False
        )
        # print("quantized_152 = ", quantized_152)
        # print("quantized_143 = ", quantized_143)

        dequantized_152 = torch.ops.torch_ipex.cast_from_fp8(
            quantized_152, scale_inv_152, dtype
        )
        dequantized_143 = torch.ops.torch_ipex.cast_from_fp8(
            quantized_143, scale_inv_143, dtype
        )
        # print("dequantized_152 = ", dequantized_152)
        # print("dequantized_143 = ", dequantized_143)

        self.assertEqual(input, dequantized_152)
        self.assertEqual(input, dequantized_143)
