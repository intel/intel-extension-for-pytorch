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
from torch.testing._internal.common_utils import TestCase


class TestFP8GEMMV2(TestCase):
    def test_fp8_linear_E5M2_float_v2(self, dtype=torch.float):
        seed = 1234
        torch.manual_seed(seed)
        is_bias = True

        input = torch.randn([8, 2], dtype=dtype, device=torch.device("xpu")) / 10
        weight = torch.rand([3, 2], dtype=dtype).xpu() / 10

        gemm_ref = torch.nn.Linear(2, 3, bias=is_bias).xpu().to(dtype)
        gemm_ref.weight.data = weight
        output_ref = gemm_ref(input)

        if is_bias:
            bias = gemm_ref.bias.data.clone()

        fp8_dtype = torch.float8_e5m2
        scale_in = (torch.ones(1) * 2).xpu()
        (torch.ones(1) * 4).xpu()
        scale_in_inv = torch.tensor([1 / 2]).xpu()
        scale_wei_inv = torch.tensor([1 / 4]).xpu()
        scale_shape = None

        input_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
            input, scale_in, False, False, fp8_dtype, scale_shape
        )
        weight_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
            weight, scale_in, False, False, fp8_dtype, scale_shape
        )

        output_fp8 = torch.ops.torch_ipex.fp8_gemm(
            input_fp8,
            False,
            weight_fp8,
            True,
            None,
            dtype,
            scale_in_inv,
            scale_wei_inv,
            bias,
            False,
        )

        # print("output_fp8 = ", output_fp8)
        # print("output_ref = ", output_ref)
        self.assertEqual(output_fp8, output_ref, rtol=1e-1, atol=1e-2)

    def test_fp8_linear_E4M3_float_v2(self, dtype=torch.float):
        seed = 1234
        torch.manual_seed(seed)
        is_bias = True

        input = torch.randn([8, 2], dtype=dtype, device=torch.device("xpu")) / 10
        weight = torch.rand([3, 2], dtype=dtype).xpu() / 10

        gemm_ref = torch.nn.Linear(2, 3, bias=is_bias).xpu().to(dtype)
        gemm_ref.weight.data = weight
        output_ref = gemm_ref(input)

        if is_bias:
            bias = gemm_ref.bias.data.clone()

        fp8_dtype = torch.float8_e4m3fn
        scale_in = (torch.ones(1) * 2).xpu()
        (torch.ones(1) * 4).xpu()
        scale_in_inv = torch.tensor([1 / 2]).xpu()
        scale_wei_inv = torch.tensor([1 / 4]).xpu()
        scale_shape = None

        input_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
            input, scale_in, False, False, fp8_dtype, scale_shape
        )
        weight_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
            weight, scale_in, False, False, fp8_dtype, scale_shape
        )

        output_fp8 = torch.ops.torch_ipex.fp8_gemm(
            input_fp8,
            False,
            weight_fp8,
            True,
            None,
            dtype,
            scale_in_inv,
            scale_wei_inv,
            bias,
            False,
        )

        # print("output_fp8 = ", output_fp8)
        # print("output_ref = ", output_ref)
        self.assertEqual(output_fp8, output_ref, rtol=1e-1, atol=1e-2)

    def test_fp8_linear_E5M2_bfloat16_v2(self, dtype=torch.bfloat16):
        seed = 1234
        torch.manual_seed(seed)
        is_bias = True

        input = torch.randn([8, 2], dtype=dtype, device=torch.device("xpu")) / 10
        weight = torch.rand([3, 2], dtype=dtype).xpu() / 10

        gemm_ref = torch.nn.Linear(2, 3, bias=is_bias).xpu().to(dtype)
        gemm_ref.weight.data = weight
        output_ref = gemm_ref(input)

        if is_bias:
            bias = gemm_ref.bias.data.clone()

        fp8_dtype = torch.float8_e5m2
        scale_in = (torch.ones(1) * 2).xpu()
        (torch.ones(1) * 4).xpu()
        scale_in_inv = torch.tensor([1 / 2]).xpu()
        scale_wei_inv = torch.tensor([1 / 4]).xpu()
        scale_shape = None

        input_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
            input, scale_in, False, False, fp8_dtype, scale_shape
        )
        weight_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
            weight, scale_in, False, False, fp8_dtype, scale_shape
        )

        output_fp8 = torch.ops.torch_ipex.fp8_gemm(
            input_fp8,
            False,
            weight_fp8,
            True,
            None,
            dtype,
            scale_in_inv,
            scale_wei_inv,
            bias,
            False,
        )
        # print("output_fp8 = ", output_fp8)
        # print("output_ref = ", output_ref)
        self.assertEqual(output_fp8, output_ref, rtol=1e-1, atol=1e-2)

    def test_fp8_linear_E4M3_bfloat16_v2(self, dtype=torch.bfloat16):
        seed = 1234
        torch.manual_seed(seed)
        is_bias = True

        input = torch.randn([8, 2], dtype=dtype, device=torch.device("xpu")) / 10
        weight = torch.rand([3, 2], dtype=dtype).xpu() / 10

        gemm_ref = torch.nn.Linear(2, 3, bias=is_bias).xpu().to(dtype)
        gemm_ref.weight.data = weight
        output_ref = gemm_ref(input)

        if is_bias:
            bias = gemm_ref.bias.data.clone()

        fp8_dtype = torch.float8_e4m3fn
        scale_in = (torch.ones(1) * 2).xpu()
        (torch.ones(1) * 4).xpu()
        scale_in_inv = torch.tensor([1 / 2]).xpu()
        scale_wei_inv = torch.tensor([1 / 4]).xpu()
        scale_shape = None

        input_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
            input, scale_in, False, False, fp8_dtype, scale_shape
        )
        weight_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
            weight, scale_in, False, False, fp8_dtype, scale_shape
        )

        output_fp8 = torch.ops.torch_ipex.fp8_gemm(
            input_fp8,
            False,
            weight_fp8,
            True,
            None,
            dtype,
            scale_in_inv,
            scale_wei_inv,
            bias,
            False,
        )

        # print("output_fp8 = ", output_fp8)
        # print("output_ref = ", output_ref)
        self.assertEqual(output_fp8, output_ref, rtol=1e-1, atol=1e-2)

    def test_fp8_linear_E5M2_half_v2(self, dtype=torch.float16):
        seed = 1234
        torch.manual_seed(seed)
        is_bias = True

        input = torch.randn([8, 2], dtype=dtype, device=torch.device("xpu")) / 10
        weight = torch.rand([3, 2], dtype=dtype).xpu() / 10

        gemm_ref = torch.nn.Linear(2, 3, bias=is_bias).xpu().to(dtype)
        gemm_ref.weight.data = weight
        output_ref = gemm_ref(input)

        if is_bias:
            bias = gemm_ref.bias.data.clone()

        fp8_dtype = torch.float8_e5m2
        scale_in = (torch.ones(1) * 2).xpu()
        (torch.ones(1) * 4).xpu()
        scale_in_inv = torch.tensor([1 / 2]).xpu()
        scale_wei_inv = torch.tensor([1 / 4]).xpu()
        scale_shape = None

        input_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
            input, scale_in, False, False, fp8_dtype, scale_shape
        )
        weight_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
            weight, scale_in, False, False, fp8_dtype, scale_shape
        )

        output_fp8 = torch.ops.torch_ipex.fp8_gemm(
            input_fp8,
            False,
            weight_fp8,
            True,
            None,
            dtype,
            scale_in_inv,
            scale_wei_inv,
            bias,
            False,
        )

        # print("output_fp8 = ", output_fp8)
        # print("output_ref = ", output_ref)
        self.assertEqual(output_fp8, output_ref, rtol=1e-1, atol=1e-2)

    def test_fp8_linear_E4M3_half_v2(self, dtype=torch.float16):
        seed = 1234
        torch.manual_seed(seed)
        is_bias = True

        input = torch.randn([8, 2], dtype=dtype, device=torch.device("xpu")) / 10
        weight = torch.rand([3, 2], dtype=dtype).xpu() / 10

        gemm_ref = torch.nn.Linear(2, 3, bias=is_bias).xpu().to(dtype)
        gemm_ref.weight.data = weight
        output_ref = gemm_ref(input)

        if is_bias:
            bias = gemm_ref.bias.data.clone()

        fp8_dtype = torch.float8_e4m3fn
        scale_in = (torch.ones(1) * 2).xpu()
        (torch.ones(1) * 4).xpu()
        scale_in_inv = torch.tensor([1 / 2]).xpu()
        scale_wei_inv = torch.tensor([1 / 4]).xpu()
        scale_shape = None

        input_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
            input, scale_in, False, False, fp8_dtype, scale_shape
        )
        weight_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
            weight, scale_in, False, False, fp8_dtype, scale_shape
        )

        output_fp8 = torch.ops.torch_ipex.fp8_gemm(
            input_fp8,
            False,
            weight_fp8,
            True,
            None,
            dtype,
            scale_in_inv,
            scale_wei_inv,
            bias,
            False,
        )

        # print("output_fp8 = ", output_fp8)
        # print("output_ref = ", output_ref)
        self.assertEqual(output_fp8, output_ref, rtol=1e-1, atol=1e-2)
