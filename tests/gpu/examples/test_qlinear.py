import torch
import torch_ipex
from torch.nn.modules.utils import _pair

from torch.testing._internal.common_utils import TestCase

class  TestTorchMethod(TestCase):
    def test_qlinear(self, dtype=torch.float):
        zero_point = 0
        
        dtype_inputs = torch.quint8
        dtype_filters = torch.qint8
        
        inputs = torch.randn(5,5)
        filters = torch.randn(5,5)
        bias = torch.randn(5)
        
        q_inputs = torch.quantize_per_tensor(inputs, 0.4, zero_point, dtype_inputs)
        q_filters = torch.quantize_per_tensor(filters, 0.5, zero_point, dtype_filters)
        
        packed_params = torch.ops.quantized.linear_prepack(q_filters, bias)
        output_int8 = torch.ops.quantized.linear(q_inputs, packed_params, 4.0, 0)
        
        inputs_gpu = inputs.to("dpcpp")
        filters_gpu = filters.to("dpcpp")
        bias_gpu = bias.to("dpcpp")
        
        q_inputs_gpu = torch.quantize_per_tensor(inputs_gpu, 0.4, zero_point, dtype_inputs)
        q_filters_gpu = torch.quantize_per_tensor(filters_gpu, 0.5, zero_point, dtype_filters)
        
        packed_params_gpu = torch.ops.quantized.linear_prepack(q_filters_gpu, bias_gpu)
        output_gpu_int8 =  torch.ops.quantized.linear(q_inputs_gpu, packed_params_gpu, 4.0, 0)

        #Align with FBGEMM, which output with UInt8.
        output_gpu_int8 =  torch.quantize_per_tensor(output_gpu_int8, 4.0, zero_point, torch.quint8)
        
        self.assertEqual(output_int8, output_gpu_int8)
