import torch
from torch.testing._internal.common_utils import TestCase

import math
import numpy as np
import intel_extension_for_pytorch  # noqa
import pytest
import platform


def _calculate_dynamic_qparams(X, dtype, reduce_range=False):
    """Calculate the dynamic quantization parameters (scale, zero_point)
    according to the min and max element of the tensor"""
    if isinstance(X, torch.Tensor):
        X = X.cpu().data.numpy()
    if dtype == torch.qint8:
        qmin, qmax = -128, 127
    else:  # dtype == torch.quint8
        qmin, qmax = 0, 255

    min_val = X.min().astype(dtype=np.float32)
    max_val = X.max().astype(dtype=np.float32)
    min_val = min(0.0, min_val)
    max_val = max(0.0, max_val)
    scale = (np.float64(max_val) - min_val) / (qmax - qmin)
    if scale == 0.0 or math.isinf(1.0 / scale):
        scale = np.float64(0.1)
        zero_point = 0

    zero_point_from_min = qmin - min_val / float(scale)
    zero_point_from_max = qmax - max_val / float(scale)
    zero_point_from_min_error = abs(qmin) - abs(min_val / float(scale))
    zero_point_from_max_error = abs(qmax) - abs(max_val / float(scale))
    if zero_point_from_min_error < zero_point_from_max_error:
        initial_zero_point = zero_point_from_min
    else:
        initial_zero_point = zero_point_from_max
    nudged_zero_point = 0

    if initial_zero_point < qmin:
        nudged_zero_point = qmin
    elif initial_zero_point > qmax:
        nudged_zero_point = qmax
    else:
        nudged_zero_point = int(round(initial_zero_point))

    return [scale.astype(np.float32), int(nudged_zero_point)]


class TestTorchMethod(TestCase):
    def test_quantize_per_tensor(self, dtype=torch.float):
        zp_vec = [0] if platform.system() == "Windows" else [0, 2]
        for data_type in [torch.qint8, torch.quint8]:
            for zp in zp_vec:
                src_cpu = torch.randn(1, 3, 2, 2)
                src_gpu = src_cpu.to("xpu")

                tensor_scale = 0.3
                print(f"\ndtpye: {data_type} sc: {tensor_scale}, zp: {zp}")

                dst_cpu = torch.quantize_per_tensor(
                    src_cpu, scale=tensor_scale, zero_point=zp, dtype=data_type
                )
                dst_gpu = torch.quantize_per_tensor(
                    src_gpu, scale=tensor_scale, zero_point=zp, dtype=data_type
                )

                print("dst cpu:", dst_cpu.int_repr())
                print("dst gpu:", dst_gpu.int_repr())
                self.assertEqual(dst_cpu, dst_gpu.cpu())

                src_cpu = torch.randn(1, 3, 2, 2)
                src_gpu = src_cpu.clone().to("xpu")
                scale_cpu = torch.tensor(tensor_scale)
                scale_gpu = scale_cpu.clone().to("xpu")
                zero_point_cpu = torch.tensor(zp)
                zero_point_gpu = zero_point_cpu.clone().to("xpu")
                dst_cpu = torch.quantize_per_tensor(
                    src_cpu, scale_cpu, zero_point_cpu, dtype=data_type
                )
                dst_gpu = torch.quantize_per_tensor(
                    src_gpu, scale_gpu, zero_point_gpu, dtype=data_type
                )

                self.assertEqual(dst_cpu, dst_gpu)

    def test_quantize_per_tensor_dynamic(self, dtype=torch.float):
        # test refer to torch/test/quantization/core/test_quantized_tensor.py:200
        # result of quantize_per_tensor_dynamic not equal cpu path but meet its function.
        for dynamic_dtype in [torch.qint8, torch.quint8]:
            max_tensor_order = 4
            max_dim_sz = 20
            num_dim = np.random.randint(low=1, high=max_tensor_order)
            dims = np.random.randint(low=1, high=max_dim_sz, size=num_dim)
            mat2quant = torch.randn(
                *dims, dtype=torch.float, device=torch.device("xpu")
            )
            result = torch.quantize_per_tensor_dynamic(mat2quant, dynamic_dtype, False)
            scale, zero_pt = _calculate_dynamic_qparams(mat2quant, dynamic_dtype, False)
            result_non_dynam = torch.quantize_per_tensor(
                mat2quant, scale, zero_pt, dynamic_dtype
            )
            self.assertEqual(result_non_dynam.cpu(), result.cpu())

    def test_quantize_tensor_channels_last(self, dtype=torch.float):
        zp_vec = [0] if platform.system() == "Windows" else [0, 2]
        for data_type in [torch.qint8, torch.quint8]:
            for tensor_zero_point in zp_vec:
                src_cpu = torch.randn(1, 3, 2, 2)
                src_gpu = src_cpu.to("xpu")
                tensor_scale = 0.3

                dst_cpu = torch.quantize_per_tensor(
                    src_cpu,
                    scale=tensor_scale,
                    zero_point=tensor_zero_point,
                    dtype=data_type,
                )
                dst_gpu = torch.quantize_per_tensor(
                    src_gpu,
                    scale=tensor_scale,
                    zero_point=tensor_zero_point,
                    dtype=data_type,
                ).to(memory_format=torch.channels_last)

                self.assertEqual(
                    True, dst_gpu.is_contiguous(memory_format=torch.channels_last)
                )
                self.assertEqual(dst_cpu, dst_gpu)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_quantize_per_channel(self, dtype=torch.float):
        src_cpu = torch.randn(1, 3, 2, 2)
        src_gpu = src_cpu.to("xpu")

        data_type = torch.qint8
        channel_scale_cpu = torch.Tensor([0.1, 0.3, 0.5])
        channel_zero_point_cpu = torch.tensor([0, 0, 0])
        channel_scale_xpu = torch.Tensor([0.1, 0.3, 0.5]).to("xpu")
        channel_zero_point_xpu = torch.tensor([0, 0, 0]).to("xpu")

        dst_cpu = torch.quantize_per_channel(
            src_cpu,
            scales=channel_scale_cpu,
            zero_points=channel_zero_point_cpu,
            dtype=data_type,
            axis=1,
        )
        dst_gpu = torch.quantize_per_channel(
            src_gpu,
            scales=channel_scale_xpu,
            zero_points=channel_zero_point_xpu,
            dtype=data_type,
            axis=1,
        )

        self.assertEqual(torch.dequantize(dst_cpu), torch.dequantize(dst_gpu))
