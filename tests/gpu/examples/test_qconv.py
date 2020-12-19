import torch
import torch_ipex
from torch.nn.modules.utils import _pair
import pytest

from torch.testing._internal.common_utils import TestCase

class  TestTorchMethod(TestCase):
    @pytest.mark.skipif(
      not 'fbgemm' in torch.backends.quantized.supported_engines,
      reason="No qengine found. USE_FBGEMM=1 is needed for building pytorch")
    def test_qconv(self, dtype=torch.float):

        print("Please open FBGEMM (PyTorch CPU INT8 default engine ) when build PyTorch, "
              "which will be referenced by GPU. Or you may meet runtime error like "
              "'Didn't find engine for operation quantized::conv2d_prepack'.")

        zero_point = 0
        
        dtype_inputs = torch.quint8
        dtype_filters = torch.qint8

        scale_in = 0.4
        scale_weight = 0.5
        scale_out = 4.0
        scale_out_2 = 8.0

        inputs = torch.randn(1,2,5,5)
        filters = torch.randn(4,2,3,3)
        bias = torch.randn(4)

        q_inputs = torch.quantize_per_tensor(inputs, scale_in, zero_point, dtype_inputs)
        q_filters = torch.quantize_per_tensor(filters, scale_weight, zero_point, dtype_filters)
        
        packed_params = torch.ops.quantized.conv2d_prepack(q_filters, bias, _pair(1),_pair(0),_pair(1),1)
        output_int8 = torch.ops.quantized.conv2d_relu(q_inputs, packed_params,_pair(1),_pair(0),_pair(1),1,scale_out,zero_point)
        output_int8_2 = torch.ops.quantized.conv2d_relu(q_inputs, packed_params,_pair(1),_pair(0),_pair(1),1,scale_out_2,zero_point)
        output_int8_3 = torch.ops.quantized.conv2d_relu(q_inputs, packed_params,_pair(1),_pair(0),_pair(1),1,scale_out_2,zero_point)
        
        inputs_gpu = inputs.to("xpu")
        filters_gpu = filters.to("xpu")
        bias_gpu = bias.to("xpu")
        
        q_inputs_gpu = torch.quantize_per_tensor(inputs_gpu, scale_in, zero_point, dtype_inputs)
        q_filters_gpu = torch.quantize_per_tensor(filters_gpu, scale_weight, zero_point, dtype_filters)
        
        packed_params_gpu = torch.ops.quantized.conv2d_prepack(q_filters_gpu, bias_gpu, _pair(1),_pair(0),_pair(1),1)
        output_gpu_int8 =  torch.ops.quantized.conv2d_relu(q_inputs_gpu, packed_params_gpu, scale_out,zero_point)
        output_gpu_int8_2 =  torch.ops.quantized.conv2d_relu(q_inputs_gpu, packed_params_gpu, scale_out_2,zero_point)
        output_gpu_int8_3 =  torch.ops.quantized.conv2d_relu(q_inputs_gpu, packed_params_gpu, scale_out_2,zero_point)

        cpu_result = torch.dequantize(output_int8)
        gpu_result = torch.dequantize(output_gpu_int8)

        cpu_result_2 = torch.dequantize(output_int8_2)
        gpu_result_2 = torch.dequantize(output_gpu_int8_2)

        cpu_result_3 = torch.dequantize(output_int8_3)
        gpu_result_3 = torch.dequantize(output_gpu_int8_3)

        self.assertEqual(cpu_result, gpu_result)
        self.assertEqual(cpu_result_2, gpu_result_2)
        self.assertEqual(cpu_result_3, gpu_result_3)
