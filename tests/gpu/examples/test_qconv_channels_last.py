import torch
import ipex
from torch.nn.modules.utils import _pair
import pytest

from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    @pytest.mark.skipif('fbgemm' not in torch.backends.quantized.supported_engines, reason="No qengine found. USE_FBGEMM=1 is needed for building pytorch")
    def test_qconv_simple_channels_last(self, dtype=torch.float):
        zero_point = 0
        dtype_inputs = torch.quint8
        dtype_filters = torch.qint8
        scale_in = 0.4
        scale_weight = 0.5
        scale_out = 4.0
        scale_out_2 = 8.0

        inputs = torch.randn(1, 2, 5, 5)
        filters = torch.randn(4, 2, 3, 3)
        bias = torch.randn(4)

        q_inputs = torch.quantize_per_tensor(inputs, scale_in, zero_point, dtype_inputs)
        q_filters = torch.quantize_per_tensor(filters, scale_weight, zero_point, dtype_filters)

        packed_params = torch.ops.quantized.conv2d_prepack(q_filters, bias, _pair(1), _pair(0), _pair(1), 1)
        output_int8 = torch.ops.quantized.conv2d_relu(q_inputs, packed_params, _pair(1), _pair(0), _pair(1), 1, scale_out, zero_point)

        inputs = inputs.to(memory_format=torch.channels_last)
        inputs_gpu = inputs.to("xpu")
        filters_gpu = filters.to("xpu")
        bias_gpu = bias.to("xpu")

        q_inputs_gpu = torch.quantize_per_tensor(inputs_gpu, scale_in, zero_point, dtype_inputs)
        q_filters_gpu = torch.quantize_per_tensor(filters_gpu, scale_weight, zero_point, dtype_filters)

        packed_params_gpu = torch.ops.quantized.conv2d_prepack(q_filters_gpu, bias_gpu, _pair(1), _pair(0), _pair(1), 1)
        output_gpu_int8 = torch.ops.quantized.conv2d_relu(q_inputs_gpu, packed_params_gpu, scale_out, zero_point)

        cpu_result = torch.dequantize(output_int8)
        gpu_result = torch.dequantize(output_gpu_int8).cpu().contiguous()

        print(cpu_result)
        print(gpu_result)

        self.assertEqual(cpu_result, gpu_result)

    @pytest.mark.skipif('fbgemm' not in torch.backends.quantized.supported_engines, reason="No qengine found. USE_FBGEMM=1 is needed for building pytorch")
    def test_qconv_simple_channels_last_3d(self, dtype=torch.float):
        zero_point = 0
        dtype_inputs = torch.quint8
        dtype_filters = torch.qint8
        scale_in = 0.4
        scale_weight = 0.5
        scale_out = 4.0
        scale_out_2 = 8.0

        inputs = torch.randn(1, 2, 5, 5, 5)
        filters = torch.randn(4, 2, 3, 3, 3)
        bias = torch.randn(4)

        q_inputs = torch.quantize_per_tensor(inputs, scale_in, zero_point, dtype_inputs)
        q_filters = torch.quantize_per_tensor(filters, scale_weight, zero_point, dtype_filters)

        stride = (1, 1, 1)
        padding = (0, 0, 0)
        dilation = (1, 1, 1)
        packed_params = torch.ops.quantized.conv3d_prepack(q_filters, bias, stride, padding, dilation, 1)
        output_int8 = torch.ops.quantized.conv3d_relu(q_inputs, packed_params, stride, padding, dilation, 1, scale_out, zero_point)

        inputs = inputs.to(memory_format=torch.channels_last_3d)
        inputs_gpu = inputs.to("xpu")
        filters_gpu = filters.to("xpu")
        bias_gpu = bias.to("xpu")
        q_inputs_gpu = torch.quantize_per_tensor(inputs_gpu, scale_in, zero_point, dtype_inputs)
        q_filters_gpu = torch.quantize_per_tensor(filters_gpu, scale_weight, zero_point, dtype_filters)

        packed_params_gpu = torch.ops.quantized.conv3d_prepack(q_filters_gpu, bias_gpu, stride, padding, dilation, 1)
        output_gpu_int8 = torch.ops.quantized.conv3d_relu(q_inputs_gpu, packed_params_gpu, scale_out, zero_point)

        cpu_result = torch.dequantize(output_int8)
        gpu_result = torch.dequantize(output_gpu_int8).cpu().contiguous()

        print(cpu_result)
        print(gpu_result)

        self.assertEqual(cpu_result, gpu_result)
