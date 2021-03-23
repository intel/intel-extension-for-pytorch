import torch
import torch_ipex

from torch.testing._internal.common_utils import TestCase

class  TestTorchMethod(TestCase):
    def test_quantize_per_tensor(self, dtype=torch.float):
        src_cpu=torch.randn(1,3,2,2)
        src_gpu = src_cpu.to("xpu")

        data_type = torch.qint8
        tensor_scale = 0.3
        tensor_zero_point = 0

        dst_cpu = torch.quantize_per_tensor(src_cpu, scale=tensor_scale, zero_point=tensor_zero_point, dtype=data_type)
        dst_gpu = torch.quantize_per_tensor(src_gpu, scale=tensor_scale, zero_point=tensor_zero_point, dtype=data_type)

        self.assertEqual(dst_cpu, dst_gpu)

    def test_quantize_per_channel(self, dtype=torch.float):
        src_cpu=torch.randn(1,3,2,2)
        src_gpu = src_cpu.to("xpu")

        data_type = torch.quint8
        channel_scale = torch.Tensor([0.1, 0.3, 0.5])
        channel_zero_point = torch.tensor([0, 0, 0])

        dst_cpu = torch.quantize_per_channel(src_cpu, scales=channel_scale, zero_points=channel_zero_point, dtype=data_type, axis=1)
        dst_gpu = torch.quantize_per_channel(src_gpu, scales=channel_scale, zero_points=channel_zero_point, dtype=data_type, axis=1)

        self.assertEqual(dst_cpu, dst_gpu)
