import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestTorchMethod(TestCase):
    def test_quantize_per_tensor(self, dtype=torch.float):
        src_cpu = torch.randn(1, 3, 2, 2)
        src_gpu = src_cpu.to("xpu")

        data_type = torch.qint8
        tensor_scale = 0.3
        tensor_zero_point = 0

        dst_cpu = torch.quantize_per_tensor(src_cpu, scale=tensor_scale, zero_point=tensor_zero_point, dtype=data_type)
        dst_gpu = torch.quantize_per_tensor(src_gpu, scale=tensor_scale, zero_point=tensor_zero_point, dtype=data_type)

        self.assertEqual(dst_cpu, dst_gpu.cpu())

        src_cpu = torch.randn(1, 3, 2, 2)
        src_gpu = src_cpu.clone().to('xpu')
        scale_cpu = torch.tensor(0.1)
        scale_gpu = scale_cpu.clone().to('xpu')
        zero_point_cpu = torch.tensor(0)
        zero_point_gpu = zero_point_cpu.clone().to('xpu')
        dst_cpu = torch.quantize_per_tensor(src_cpu, scale_cpu, zero_point_cpu, dtype=data_type)
        dst_gpu = torch.quantize_per_tensor(src_gpu, scale_gpu, zero_point_gpu, dtype=data_type)

        self.assertEqual(dst_cpu, dst_gpu)

    def test_quantize_tensor_channels_last(self, dtype=torch.float):
        src_cpu = torch.randn(1, 3, 2, 2)
        src_gpu = src_cpu.to("xpu")

        data_type = torch.qint8
        tensor_scale = 0.3
        tensor_zero_point = 0

        dst_cpu = torch.quantize_per_tensor(src_cpu, scale=tensor_scale, zero_point=tensor_zero_point, dtype=data_type)
        dst_gpu = torch.quantize_per_tensor(src_gpu, scale=tensor_scale, zero_point=tensor_zero_point,
                                            dtype=data_type).to(memory_format=torch.channels_last)

        self.assertEqual(True, dst_gpu.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(dst_cpu, dst_gpu)

    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_quantize_per_channel(self, dtype=torch.float):
        src_cpu = torch.randn(1, 3, 2, 2)
        src_gpu = src_cpu.to("xpu")

        data_type = torch.qint8
        channel_scale_cpu = torch.Tensor([0.1, 0.3, 0.5])
        channel_zero_point_cpu = torch.tensor([0, 0, 0])
        channel_scale_xpu = torch.Tensor([0.1, 0.3, 0.5]).to("xpu")
        channel_zero_point_xpu = torch.tensor([0, 0, 0]).to("xpu")

        dst_cpu = torch.quantize_per_channel(src_cpu, scales=channel_scale_cpu,
                                             zero_points=channel_zero_point_cpu, dtype=data_type, axis=1)
        dst_gpu = torch.quantize_per_channel(src_gpu, scales=channel_scale_xpu,
                                             zero_points=channel_zero_point_xpu, dtype=data_type, axis=1)

        self.assertEqual(torch.dequantize(dst_cpu), torch.dequantize(dst_gpu))
