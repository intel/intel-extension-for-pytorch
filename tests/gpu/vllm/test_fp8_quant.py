# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import intel_extension_for_pytorch as ipex
from typing import Optional, Union
import random
import numpy as np


def as_float32_tensor(x: Union[float, torch.tensor]) -> torch.tensor:
    return torch.as_tensor(x, dtype=torch.float32, device='xpu')

def ref_dynamic_per_tensor_fp8_quant(x, fp8_dtype):

    fp8_traits = torch.finfo(fp8_dtype)
    fp8_traits_max = fp8_traits.max
    fp8_traits_min = fp8_traits.min
    fp8_max = as_float32_tensor(fp8_traits_max)
    one = as_float32_tensor(1.0)

    # For fp8, in order to match the xpu kernel output, we have to do exactly
    # the same operations as in the corresponding fp8 kernel to prevent
    # rounding errors.

    x_max = as_float32_tensor(x.abs().max())
    ref_scale = x_max / fp8_max
    ref_iscale = one / ref_scale
    ref_out = (as_float32_tensor(x) * ref_iscale).clamp(
        fp8_traits_min, fp8_traits_max).to(fp8_dtype)
    return ref_out, ref_scale.view((1, ))

def seed_everything(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

DTYPES = [torch.half, torch.bfloat16, torch.float]
HIDDEN_SIZES = [1, 2, 3, 4, 16, 67, 768, 2048, 5120, 5137, 8192,
                8193]  # Arbitrary values for testing
HIDDEN_SIZES += list(range(1024, 1033))  # vectorized conversion edge cases
NUM_TOKENS = [1, 7, 83, 4096]  # Arbitrary values for testing
SCALE_UBS = [True, False]
SEEDS = [0]
FP8_DTYPES = [torch.float8_e5m2, torch.float8_e4m3fn]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@torch.inference_mode()
def test_dynamic_per_tensor_fp8_quant(num_tokens: int, hidden_size: int,
                                      dtype: torch.dtype, seed: int, fp8_dtype: torch.dtype) -> None:
    seed_everything(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="xpu")

    ref_out, ref_scale = ref_dynamic_per_tensor_fp8_quant(x, fp8_dtype)

    ops_out, ops_scale = ipex.llm.quantization.IPEXFP8ScaledQuant.scaled_fp8_quant(x, fp8_dtype)

    torch.testing.assert_close(ref_scale, ops_scale)
    torch.testing.assert_close(ref_out.to(dtype=torch.float32),
                               ops_out.to(dtype=torch.float32))


# Regression test for a case with large activations where an int32 index cannot
# represent the number of elements.
@torch.inference_mode()
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
def test_fp8_quant_large(seed: int, fp8_dtype: torch.dtype) -> None:
    seed_everything(seed)

    num_tokens = 1024000  # Mistral-Nemo's max_position_embeddings
    hidden_size = 1152  # Smallest hidden_size to reproduce the error
    dtype = torch.bfloat16

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="xpu")
    ref_out, scale = ref_dynamic_per_tensor_fp8_quant(x, fp8_dtype)

    ops_out, _ = ipex.llm.quantization.IPEXFP8ScaledQuant.scaled_fp8_quant(x, fp8_dtype, scale)

    # Minimize memory footprint in this test by freeing x and upconverting
    # the outputs in place. (torch.allclose does not support fp8)
    del x
    ref_out = ref_out.to(dtype=dtype)
    ops_out = ops_out.to(dtype=dtype)

    torch.testing.assert_close(ref_out, ops_out)

if __name__ == "__main__":
    test_dynamic_per_tensor_fp8_quant(1024, 1024, torch.float16, 0)
