from itertools import product
import math
import random
import time
import numpy as np
import pytest
import torch
from torch import Tensor
import intel_extension_for_pytorch as ipex

torch.set_printoptions(precision=5, sci_mode=False, linewidth=120, edgeitems=20, threshold=10000)
k = 20

def percentile_clipping(grad: Tensor, gnorm_vec: Tensor, step: int, percentile: int = 5):
    """Applies percentile clipping

    grad: torch.Tensor
        The gradient tensor.
    gnorm_vec: torch.Tensor
        Vector of gradient norms. 100 elements expected.
    step: int
        The current optimiation steps (number of past gradient norms).

    """

    if grad.dtype == torch.float32:
        ipex.xpu.bitsandbytes.cpercentile_clipping_g32(
            grad,
            gnorm_vec,
            step,
            grad.numel()
        )
    elif grad.dtype == torch.float16:
        ipex.xpu.bitsandbytes.cpercentile_clipping_g16(
            grad,
            gnorm_vec,
            step,
            grad.numel()
        )
    else:
        raise ValueError(f"Gradient type {grad.dtype} not supported!")

    current_gnorm = torch.sqrt(gnorm_vec[step % 100])
    vals, idx = torch.sort(gnorm_vec)
    clip_value = torch.sqrt(vals[percentile])
    gnorm_scale = 1.0

    if current_gnorm > clip_value:
        gnorm_scale = clip_value / current_gnorm

    return current_gnorm, clip_value, gnorm_scale

@pytest.mark.parametrize("gtype", [torch.float32, torch.float16], ids=["float", "half"])
def test_percentile_clipping(gtype):
    gnorm_vec1 = torch.zeros(100, device="xpu")
    gnorm_vec2 = torch.zeros(100, device="xpu")
    n = 4
    step = 0
    percentile = 5
    for i in range(k):
        step += 1
        g = torch.randn(n, n, dtype=gtype, device="xpu")
        gnorm1, clip2, gnorm_scale = percentile_clipping(g, gnorm_vec2, step, percentile=percentile)
        assert gnorm_scale == 1.0 if gnorm1 < clip2 else clip2 / gnorm1

        gnorm2 = torch.norm(g.float())
        if step == 1:
            gnorm_vec1[:] = gnorm2
        else:
            gnorm_vec1[step % 100] = gnorm2

        vals, idx = torch.sort(gnorm_vec1)
        clip1 = vals[percentile]

        torch.testing.assert_close(gnorm_vec1, torch.sqrt(gnorm_vec2))
        torch.testing.assert_close(clip1, clip2)
        torch.testing.assert_close(gnorm1, gnorm2)

if __name__ == "__main__":
    test_percentile_clipping(torch.float32)
    test_percentile_clipping(torch.float16)