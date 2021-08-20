import torch
from torch.testing._internal.common_utils import TestCase
import ipex

class TestTorchMethod(TestCase):
    def test_cat_array(self, dtype=torch.float):
        zero_point = 0
        dtype_inputs = torch.quint8
        
        input1 = torch.randn(1,1,5,5)
        input2 = torch.randn(1,1,5,5)
        input3 = torch.randn(1,1,5,5)

        input1_gpu = input1.to("xpu")
        input2_gpu = input2.to("xpu")
        input3_gpu = input3.to("xpu")
        
        q_input1 = torch.quantize_per_tensor(input1, 0.4, zero_point, dtype_inputs)
        q_input2 = torch.quantize_per_tensor(input2, 0.5, zero_point, dtype_inputs)
        q_input3 = torch.quantize_per_tensor(input3, 0.6, zero_point, dtype_inputs)
        
        output_int8 = torch.ops.quantized.cat([q_input1, q_input2, q_input3], dim=1, scale=0.2, zero_point=0)
        
        q_input1_gpu = torch.quantize_per_tensor(input1_gpu, 0.4, zero_point, dtype_inputs)
        q_input2_gpu = torch.quantize_per_tensor(input2_gpu, 0.5, zero_point, dtype_inputs)
        q_input3_gpu = torch.quantize_per_tensor(input3_gpu, 0.6, zero_point, dtype_inputs)
        
        output_gpu_int8 =  torch.ops.quantized.cat([q_input1_gpu, q_input2_gpu, q_input3_gpu], dim=1, scale=0.2, zero_point=0)
       
        self.assertEqual(output_int8, output_gpu_int8)

