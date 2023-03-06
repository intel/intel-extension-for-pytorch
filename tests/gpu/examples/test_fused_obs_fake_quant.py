import math
import torch
import torch.nn as nn
from typing import Tuple
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


# Ported from quantization/core/test_workflow_ops.py
def _get_scale_zp(
        min_val: float,
        max_val: float,
        dtype: torch.dtype,
        reduce_range: bool = False,
        preserve_sparsity: bool = False) -> Tuple[float, int]:
    """
    Calculate the quantization parameters (scale, zero_point)
    based on the min and max element of the tensor
    """
    if dtype == torch.qint8:
        if reduce_range:
            qmin, qmax = -64, 63
        else:
            qmin, qmax = -128, 127
    else:
        if reduce_range:
            qmin, qmax = 0, 127
        else:
            qmin, qmax = 0, 255

    if min_val < 0 and max_val > 0 and preserve_sparsity:
        symmetric_qmin = int(-((qmax - qmin) / 2 + 1))
        symmetric_qmax = int((qmax - qmin) / 2)
        max_scale = max(
            abs(min_val / symmetric_qmin), abs(max_val / symmetric_qmax)
        )
        min_val = max_scale * symmetric_qmin
        max_val = max_scale * symmetric_qmax
    min_val = min(min_val, 0.0)
    max_val = max(max_val, 0.0)
    scale = (max_val - min_val) / (qmax - qmin)
    if scale == 0.0 or math.isinf(1.0 / scale):
        scale = 0.1
        zero_point = 0

    zero_point_from_min = qmin - min_val / float(scale)
    zero_point_from_max = qmax - max_val / float(scale)
    zero_point_from_min_error = abs(qmin) - abs(min_val / float(scale))
    zero_point_from_max_error = abs(qmax) - abs(max_val / float(scale))
    if zero_point_from_min_error < zero_point_from_max_error:
        initial_zero_point = zero_point_from_min
    else:
        initial_zero_point = zero_point_from_max

    if min_val < 0 and max_val > 0 and preserve_sparsity:
        initial_zero_point = (qmin + qmax) / 2 + 1

    nudged_zero_point = 0

    if initial_zero_point < qmin:
        nudged_zero_point = qmin
    elif initial_zero_point > qmax:
        nudged_zero_point = qmax
    else:
        nudged_zero_point = int(round(initial_zero_point))

    return (scale, int(nudged_zero_point))


def _get_tensor_min_max(
    X: torch.Tensor,
    running_min: float = float("inf"),
    running_max: float = float("-inf"),
    averaging_const: float = 0.01) -> Tuple[float, float]:

    min_val = X.min().to(dtype=torch.float32).item()
    max_val = X.max().to(dtype=torch.float32).item()

    if not math.isinf(running_min):
        min_val = running_min + averaging_const * (min_val - running_min)
    if not math.isinf(running_max):
        max_val = running_max + averaging_const * (max_val - running_max)

    return min_val, max_val


# Reference method for fake quantize
# Note: because scale/zero_point are left as float in the actual kernel, this mimics how fake_quant works for float16/64
def _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, quant_min, quant_max):
    dtype = X.dtype
    res = ((torch.clamp(torch.round(X.to(torch.float32) * (1.0 / scale) + zero_point), quant_min, quant_max) - zero_point) * scale)
    return res.to(dtype)


class TestNNMethod(TestCase):
    def test_fused_obs_fake_quant_moving_avg(self, device=dpcpp_device, symmetric_quant=True) -> None:
        """
        Tests the case where we call the fused_obs_fake_quant op multiple times
        and update the running_min and max of the activation tensors.
        """
        in_running_min_ref = out_running_min_ref = float("inf")
        in_running_min_op = torch.tensor(float("inf"), device=device)
        in_running_max_ref = out_running_max_ref = float("-inf")
        in_running_max_op = torch.tensor(float("-inf"), device=device)
        avg_const = 0.01
        scale = torch.tensor([1.0], device=device)
        zero_point = torch.tensor([0], dtype=torch.int, device=device)
        observer_on = fake_quant_on = 0

        pt_op = torch.fused_moving_avg_obs_fake_quant
        # enable observer after 2 iterations and fake_quant after 4 iterations
        for i in range(10):
            if i > 2:
                observer_on = 1
            if i > 4:
                fake_quant_on = 1

            x = torch.randn(5, 5, device=device)
            out = pt_op(
                x,
                torch.tensor(observer_on, device=device),
                torch.tensor(fake_quant_on, device=device),
                in_running_min_op,
                in_running_max_op,
                scale,
                zero_point,
                avg_const,
                0,
                255,
                0,
                False,
                symmetric_quant,
            )
            if observer_on:
                (
                    in_running_min_ref,
                    in_running_max_ref,
                ) = _get_tensor_min_max(
                    x,
                    running_min=in_running_min_ref,
                    running_max=in_running_max_ref,
                    averaging_const=0.01,
                )

            if fake_quant_on:
                x_scale, x_zero_point = _get_scale_zp(
                    in_running_min_ref,
                    in_running_max_ref,
                    torch.quint8,
                    preserve_sparsity=symmetric_quant,
                )
                x_in = _fake_quantize_per_tensor_affine_reference(
                    x, x_scale, x_zero_point, 0, 255
                )
                self.assertEqual(scale, x_scale)
                self.assertEqual(zero_point, x_zero_point)
            else:
                x_in = x

            self.assertEqual(in_running_min_ref, in_running_min_op)
            self.assertEqual(in_running_max_ref, in_running_max_op)
            torch.testing.assert_allclose(out, x_in)

        # Test empty input works
        x = torch.empty(0, 5, device=device)
        out = pt_op(
            x,
            torch.tensor(1, device=device),
            torch.tensor(1, device=device),
            in_running_min_op,
            in_running_max_op,
            scale,
            zero_point,
            avg_const,
            0,
            255,
            0,
            False,
            symmetric_quant,
        )
        output_shape = (0, 5)
        self.assertEqual(out.shape, output_shape)

