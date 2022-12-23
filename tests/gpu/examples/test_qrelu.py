import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
from torch.nn.modules.utils import _pair
import intel_extension_for_pytorch # noqa
import numpy as np


class TestNNMethod(TestCase):
    def test_QReLU_s8(self, dtype=torch.float):
        dtype_inputs = torch.qint8
        scale = 0.04
        x_cpu = torch.randn([1, 1, 3, 4], device=torch.device("cpu"))
        x_gpu = x_cpu.to('xpu')
        mod = nn.ReLU()

        q_cpu = torch.quantize_per_tensor(x_cpu, scale, 0, dtype_inputs)
        y_cpu = mod(q_cpu)

        mod.to("xpu")
        q_gpu = torch.quantize_per_tensor(x_gpu, scale, 0, dtype_inputs)
        y_gpu = mod(q_gpu)

        print("y_cpu:", y_cpu)
        print("y_gpu:", y_gpu)

        self.assertEqual(torch.dequantize(y_cpu), torch.dequantize(y_gpu))

    def test_QReLU_u8(self, dtype=torch.float):
        zero_point_u8 = 128
        zero_point_s8 = 0
        dtype_inputs = torch.quint8
        dtype_filters = torch.qint8

        scale_in = 0.4
        scale_weight = 0.5
        scale_out = 0.3

        inputs = torch.randn(1, 2, 5, 5)
        filters = torch.randn(4, 2, 3, 3)
        bias = torch.randn(4)

        q_inputs = torch.quantize_per_tensor(inputs, scale_in, zero_point_u8, dtype_inputs)  # f32 / sc_in + 128  u8
        q_filters = torch.quantize_per_tensor(filters, scale_weight, zero_point_s8, dtype_filters)  # w32 / sc_wgh  s8
        packed_params = torch.ops.quantized.conv2d_prepack(q_filters, bias, _pair(1), _pair(0), _pair(1), 1)
        output_int8 = torch.ops.quantized.conv2d_relu(q_inputs, packed_params, _pair(1),
                                                      _pair(0), _pair(1), 1, scale_out, zero_point_u8)

        inputs_gpu = inputs.to("xpu")
        filters_gpu = filters.to("xpu")
        bias_gpu = bias.to("xpu")

        q_inputs_gpu = torch.quantize_per_tensor(inputs_gpu, scale_in, zero_point_u8, dtype_inputs)  # f32 / sc_in  s8
        self.assertEqual(torch.dequantize(q_inputs), torch.dequantize(q_inputs_gpu), "Input is not quantized equal")

        q_filters_gpu = torch.quantize_per_tensor(filters_gpu, scale_weight, zero_point_s8, dtype_filters)  # w32 / sc_wgh s8
        self.assertEqual(torch.dequantize(q_filters), torch.dequantize(q_filters_gpu), "Weight is not quantized equal")

        packed_params_gpu = torch.ops.quantized.conv2d_prepack(q_filters_gpu, bias_gpu, _pair(1), _pair(0), _pair(1), 1)
        output_gpu_int8 = torch.ops.quantized.conv2d_relu(q_inputs_gpu, packed_params_gpu, scale_out, zero_point_u8)

        output_gpu_int8 = torch.ops.quantized.conv2d_relu(q_inputs_gpu, packed_params_gpu, scale_out, zero_point_u8)


        mod = torch.nn.ReLU()

        res_cpu = mod(output_int8)
        res_gpu = mod(output_gpu_int8)
        print(output_int8)
        print(output_gpu_int8)
        np.testing.assert_almost_equal(torch.dequantize(res_cpu).numpy(), torch.dequantize(res_gpu).cpu().numpy(), decimal=0)
