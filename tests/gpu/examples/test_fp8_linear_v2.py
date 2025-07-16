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
import pytest
import intel_extension_for_pytorch as ipex  # noqa
from intel_extension_for_pytorch.llm.quantization.utils import QuantDtype


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("dtype", [torch.float, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_input_fp8", [True, False])
@pytest.mark.parametrize("is_bias", [True, False])
def test_fp8_linear_v2(fp8_dtype, dtype, is_input_fp8, is_bias):
    seed = 1234
    torch.manual_seed(seed)

    input = torch.randn([8, 2], dtype=dtype, device=torch.device("xpu")) / 10.0
    weight = torch.rand([3, 2], dtype=dtype).xpu() / 10.0
    gemm_ref = torch.nn.Linear(2, 3, bias=is_bias).xpu().to(dtype)

    scale_in = (torch.ones(1) * 2).xpu()
    scale_wei = (torch.ones(1) * 4).xpu()
    scale_in_inv = torch.tensor([0.5]).xpu()
    scale_wei_inv = torch.tensor([0.25]).xpu()
    scale_shape = None

    input_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
        input, scale_in, False, False, fp8_dtype, scale_shape
    )
    weight_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
        weight, scale_wei, False, False, fp8_dtype, scale_shape
    )

    weight_dequant = torch.ops.torch_ipex.cast_from_fp8(
        weight_fp8, scale_wei_inv, dtype
    )
    gemm_ref.weight.data = weight_dequant
    output_ref = gemm_ref(input)

    output_fp8 = torch.ops.torch_ipex.fp8_gemm(
        input_fp8 if is_input_fp8 else input,
        False,
        weight_fp8,
        True,
        None,
        dtype,
        scale_in_inv if is_input_fp8 else None,
        scale_wei_inv,
        gemm_ref.bias.data.clone() if is_bias else None,
        False,
    )

    torch.testing.assert_close(output_fp8, output_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_bias", [True, False])
@pytest.mark.parametrize("is_nt", [True, False])
@pytest.mark.parametrize("is_mbk", [True, False])
def test_fp8_linear_w8a16(fp8_dtype, dtype, is_bias, is_nt, is_mbk):
    seed = 1234
    torch.manual_seed(seed)

    input = torch.randn([1, 8, 2], dtype=dtype, device=torch.device("xpu")) / 10.0
    weight = torch.rand([3, 2], dtype=dtype).xpu() / 10.0

    gemm_ref = torch.nn.Linear(2, 3, bias=is_bias).xpu().to(dtype)

    scale_wei = (torch.ones(1) * 4).xpu()
    scale_wei_inv = torch.tensor([0.25]).xpu()
    scale_shape = None

    weight_fp8, _ = torch.ops.torch_ipex.cast_to_fp8(
        weight, scale_wei, False, False, fp8_dtype, scale_shape
    )

    weight_dequant = torch.ops.torch_ipex.cast_from_fp8(
        weight_fp8, scale_wei_inv, dtype
    )
    gemm_ref.weight.data = weight_dequant
    output_ref = gemm_ref(input)

    if not is_nt:
        weight_fp8 = weight_fp8.transpose(0, 1).contiguous().transpose(0, 1)

    fp8_linear = ipex.llm.quantization.IPEXWeightOnlyQuantizedLinear.from_weight(
        weight_fp8,
        scale_wei_inv,
        torch.Tensor(),
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        dtype=(
            QuantDtype.FP8_E5M2
            if fp8_dtype == torch.float8_e5m2
            else QuantDtype.FP8_E4M3FN
        ),
    )

    if is_mbk:
        input = input.transpose(0, 1)

    output_fp8 = fp8_linear(input, gemm_ref.bias.data.clone() if is_bias else None)
    output_fp8 = output_fp8.transpose(0, 1) if is_mbk else output_fp8

    torch.testing.assert_close(output_fp8, output_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test_fp8_linear_w8a16(torch.float8_e5m2, torch.float16, True, True, True)
