import numpy as np
import torch
from torch.testing._internal.common_utils import TestCase
from torch.nn.modules.utils import _pair
import intel_extension_for_pytorch  # noqa

class TestTorchMethod(TestCase):
    def test_s8s8(self, dtype=torch.float):
        a_cpu = torch.randn(1, 4, 2, 2)
        b_cpu = torch.randn(1, 4, 2, 2)
        a_gpu = a_cpu.to("xpu")
        b_gpu = b_cpu.to("xpu")

        data_type = torch.qint8
        a_max = torch.abs(a_cpu).max()
        b_max = torch.abs(b_cpu).max()
        a_scale = a_max / 127.0
        a_zero_point = 0
        b_scale = b_max / 127.0
        b_zero_point = 0
        add_cpu = a_cpu + b_cpu
        add_max = torch.abs(add_cpu).max()
        add_scale = add_max / 127.0
        add_zero_point = 0
        cpu_res = torch.nn.functional.relu(add_cpu)
        print("cpu res:\n", torch.nn.functional.relu(add_cpu))
        q_ref = torch.quantize_per_tensor(cpu_res.to("xpu"), add_scale, 128, torch.quint8)
        q_ref = torch.dequantize(q_ref)
        

        qa_gpu = torch.quantize_per_tensor(a_gpu, scale=a_scale, zero_point=a_zero_point, dtype=data_type)
        qb_gpu = torch.quantize_per_tensor(b_gpu, scale=b_scale, zero_point=b_zero_point, dtype=data_type)
        qo_gpu = torch.ops.quantized.add_relu(qa_gpu, qb_gpu, add_scale, 128)
        o_gpu = torch.dequantize(qo_gpu)
        print("gpu res:\n", o_gpu.cpu())

        qa_cpu = torch.quantize_per_tensor(a_cpu, scale=a_scale, zero_point=a_zero_point, dtype=data_type)
        qb_cpu = torch.quantize_per_tensor(b_cpu, scale=b_scale, zero_point=b_zero_point, dtype=data_type)
        qadd_cpu = torch.ops.quantized.add(qa_cpu, qb_cpu, add_scale, add_zero_point)
        qo_cpu = torch.nn.functional.relu(qadd_cpu)
        o_cpu = torch.dequantize(qo_cpu)
        print("cpu q_res:\n",o_cpu)

        np.testing.assert_almost_equal(q_ref.cpu().numpy(), o_gpu.cpu().numpy(), decimal=0)

    def test_u8u8(self, dtype=torch.float):
        zero_point_u8 = 128
        zero_point_s8 = 0
        dtype_inputs = torch.quint8
        dtype_filters = torch.qint8

        scale_in = 0.4
        scale_weight = 0.5
        scale_out = 0.3
        add_scale_out = 0.45
        add_zero_point = 128

        inputs = torch.randn(1, 2, 5, 5)
        inputs_other = torch.randn(1, 2, 5, 5)
        filters = torch.randn(4, 2, 3, 3)
        bias = torch.randn(4)

        print("start cpu computation")
        q_inputs = torch.quantize_per_tensor(inputs, scale_in, zero_point_u8, dtype_inputs)  # f32 / sc_in + 128  u8
        q_inputs_other = torch.quantize_per_tensor(inputs_other, scale_in, zero_point_u8, dtype_inputs)
        q_filters = torch.quantize_per_tensor(filters, scale_weight, zero_point_s8, dtype_filters)  # w32 / sc_wgh  s8
        packed_params = torch.ops.quantized.conv2d_prepack(q_filters, bias, _pair(1), _pair(0), _pair(1), 1)
        output_int8 = torch.ops.quantized.conv2d_relu(q_inputs, packed_params, scale_out, zero_point_u8)
        output_int8_other = torch.ops.quantized.conv2d_relu(q_inputs_other, packed_params, scale_out, zero_point_u8)

        cpu_ref = torch.ops.quantized.add_relu(output_int8, output_int8_other, add_scale_out, add_zero_point)

        print("starg xpu computation")
        inputs_gpu = inputs.to("xpu")
        inputs_other_gpu = inputs_other.to("xpu")
        filters_gpu = filters.to("xpu")
        bias_gpu = bias.to("xpu")

        q_inputs_gpu = torch.quantize_per_tensor(inputs_gpu, scale_in, zero_point_u8, dtype_inputs)  # f32 / sc_in  s8
        q_inputs_other_gpu = torch.quantize_per_tensor(inputs_other_gpu, scale_in, zero_point_u8, dtype_inputs)
        # self.assertEqual(torch.dequantize(q_inputs), torch.dequantize(q_inputs_gpu), "Input is not quantized equal")

        q_filters_gpu = torch.quantize_per_tensor(filters_gpu, scale_weight, zero_point_s8, dtype_filters)  # w32 / sc_wgh s8
        print("finishes quantize")
        # self.assertEqual(torch.dequantize(q_filters), torch.dequantize(q_filters_gpu), "Weight is not quantized equal")

        packed_params_gpu = torch.ops.quantized.conv2d_prepack(q_filters_gpu, bias_gpu, _pair(1), _pair(0), _pair(1), 1)
        output_gpu_int8 = torch.ops.quantized.conv2d_relu(q_inputs_gpu, packed_params_gpu, scale_out, zero_point_u8)
        print("finishes first conv_relu")

        output_gpu_int8_other = torch.ops.quantized.conv2d_relu(q_inputs_other_gpu, packed_params_gpu, scale_out, zero_point_u8)
        print("finished second conv relu")
        gpu_res = torch.ops.quantized.add_relu(output_gpu_int8, output_gpu_int8_other, add_scale_out, add_zero_point)
        print("finishes add_relu")

        print("cpu_res: \n", torch.dequantize(cpu_ref))
        print("gpu_res:\n", torch.dequantize(gpu_res).cpu())

        np.testing.assert_almost_equal(torch.dequantize(cpu_ref).numpy(), torch.dequantize(gpu_res).cpu().numpy(), decimal=0)
