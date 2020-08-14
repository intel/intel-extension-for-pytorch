import torch
import torch_ipex

from torch.testing._internal.common_utils import TestCase

class  TestTorchMethod(TestCase):
    def test_dequantize_per_channel(self, dtype=torch.float):
        src=torch.randn(1,3,2,2)
        src_gpu = src.to("dpcpp")

        data_type = torch.qint8

        tensor_scale = 0.3
        tensor_zero_point = 0
        
        dst_q = torch.quantize_per_tensor(src, scale=tensor_scale, zero_point=tensor_zero_point, dtype=data_type)
        dst = torch.dequantize(dst_q)

        dst_gpu_q = torch.quantize_per_tensor(src_gpu, scale=tensor_scale, zero_point=tensor_zero_point, dtype=data_type)
        dst_gpu = torch.dequantize(dst_gpu_q)
        
        self.assertEqual(dst, dst_gpu)

    def test_dequantize_per_tensor(self, dtype=torch.float):
        src=torch.randn(1,3,2,2)
        src_gpu = src.to("dpcpp")

        data_type = torch.quint8
        
        channel_scale = torch.Tensor([0.1, 0.3, 0.5])
        channel_zero_point = torch.tensor([0, 0, 0])
        
        dst_q = torch.quantize_per_channel(src, scales=channel_scale, zero_points=channel_zero_point, dtype=data_type, axis=1)
        dst = torch.dequantize(dst_q)

        dst_gpu_q = torch.quantize_per_channel(src_gpu, scales=channel_scale, zero_points=channel_zero_point, dtype=data_type, axis=1)
        dst_gpu = torch.dequantize(dst_gpu_q)
        
        self.assertEqual(dst, dst_gpu)

    def test_dequantize_FP32_input(self, dtype=torch.float):
        src=torch.randn(1,3,2,2)
        src_gpu = src.to("dpcpp")

        dst_gpu = torch.dequantize(src_gpu)
        
        self.assertEqual(src_gpu, dst_gpu)
