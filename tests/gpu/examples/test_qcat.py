import torch
from torch.nn.modules.utils import _pair

from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

class Fake_Q_Cat_Dequantize(torch.nn.Module):
    def __init__(self, dim, scale, zero_point):
        super(Fake_Q_Cat_Dequantize, self).__init__()
        self.dim = dim
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, tensors):
        cat_input = []
        for t in tensors:
            cat_input.append(torch.dequantize(t))
        output_dq = torch.cat(cat_input, dim=self.dim)
        return output_dq

class Q_Cat_Dequantize(torch.nn.Module):
    def __init__(self, dim, scale, zero_point):
        super(Q_Cat_Dequantize, self).__init__()
        self.dim = dim
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, tensors):
        output_q = torch.ops.quantized.cat(tensors, dim=self.dim, scale=self.scale, zero_point=self.zero_point)
        output_dq = torch.dequantize(output_q)
        return output_dq

class Fake_Q_Conv_Relu_Cat_Dequantize(torch.nn.Module):
    def __init__(self, dim, scale, zero_point, scal):
        super(Fake_Q_Conv_Relu_Cat_Dequantize, self).__init__()
        self.dim = dim
        self.scale = scale
        self.zero_point = zero_point
        self.scale_list = scal

    def forward(self, tensors, weights, bias):
        cat_input = []
        for i in range(len(tensors)):
            conv_out = None
            packed_params = torch.ops.quantized.conv2d_prepack(weights[i], bias[i], _pair(1), _pair(0), _pair(1), 1)
            relu_out = torch.ops.quantized.conv2d_relu(tensors[i], packed_params,
                                                       _pair(1), _pair(0), _pair(1), 1, self.scale_list[i], self.zero_point)
            cat_input.append(torch.dequantize(relu_out))
        output_dq = torch.cat(cat_input, dim=self.dim)
        return output_dq

class Q_Conv_Relu_Cat_Dequantize(torch.nn.Module):
    def __init__(self, dim, scale, zero_point, scal, is_cpu=False):
        super(Q_Conv_Relu_Cat_Dequantize, self).__init__()
        self.dim = dim
        self.scale = scale
        self.zero_point = zero_point
        self.is_cpu = is_cpu
        self.scale_list = scal

    def forward(self, tensors, weights, bias):
        cat_input = []
        for i in range(len(tensors)):
            conv_out = None
            packed_params = torch.ops.quantized.conv2d_prepack(weights[i], bias[i], _pair(1), _pair(0), _pair(1), 1)
            if self.is_cpu:
                relu_out = torch.ops.quantized.conv2d_relu(tensors[i], packed_params,
                                                           _pair(1), _pair(0), _pair(1), 1, self.scale_list[i], self.zero_point)
            else:
                relu_out = torch.ops.quantized.conv2d_relu(tensors[i], packed_params, self.scale_list[i], self.zero_point)
            cat_input.append(relu_out)
        output_q = torch.ops.quantized.cat(cat_input, dim=self.dim, scale=self.scale, zero_point=self.zero_point)
        output_dq = torch.dequantize(output_q)
        return output_dq

class Fake_Q_Conv_Cat_Dequantize(torch.nn.Module):
    def __init__(self, dim, scale, zero_point, scale_list):
        super(Fake_Q_Conv_Cat_Dequantize, self).__init__()
        self.dim = dim
        self.scale = scale
        self.zero_point = zero_point
        self.scale_list = scale_list

    def forward(self, tensors, weights, bias):
        cat_input = []
        for i in range(len(tensors)):
            conv_out = None
            packed_params = torch.ops.quantized.conv2d_prepack(weights[i], bias[i], _pair(1), _pair(0), _pair(1), 1)
            conv_out = torch.ops.quantized.conv2d(tensors[i], packed_params,
                                                  _pair(1), _pair(0), _pair(1), 1, self.scale_list[i], self.zero_point)
            cat_input.append(torch.dequantize(conv_out))
        output_dq = torch.cat(cat_input, dim=self.dim)
        return output_dq

class Q_Conv_Cat_Dequantize(torch.nn.Module):
    def __init__(self, dim, scale, zero_point, scale_list, is_cpu=False):
        super(Q_Conv_Cat_Dequantize, self).__init__()
        self.dim = dim
        self.scale = scale
        self.zero_point = zero_point
        self.is_cpu = is_cpu
        self.scale_list = scale_list

    def forward(self, tensors, weights, bias):
        cat_input = []
        for i in range(len(tensors)):
            conv_out = None
            packed_params = torch.ops.quantized.conv2d_prepack(weights[i], bias[i], _pair(1), _pair(0), _pair(1), 1)
            if self.is_cpu:
                conv_out = torch.ops.quantized.conv2d(tensors[i], packed_params,
                                                      _pair(1), _pair(0), _pair(1), 1, self.scale_list[i], self.zero_point)
            else:
                conv_out = torch.ops.quantized.conv2d(tensors[i], packed_params, self.scale_list[i], self.zero_point)
            cat_input.append(conv_out)
        output_q = torch.ops.quantized.cat(cat_input, dim=self.dim, scale=self.scale, zero_point=self.zero_point)
        output_dq = torch.dequantize(output_q)
        return output_dq

class TestTorchMethod(TestCase):
    def test_cat_array_quint8(self, dtype=torch.float):
        zero_point = 0
        dtype_inputs = torch.quint8

        input1 = torch.randn(1, 1, 5, 5)
        input2 = torch.randn(1, 1, 5, 5)
        input3 = torch.randn(1, 1, 5, 5)

        input1_gpu = input1.to("xpu")
        input2_gpu = input2.to("xpu")
        input3_gpu = input3.to("xpu")

        q_input1 = torch.quantize_per_tensor(input1, 0.4, zero_point, dtype_inputs)
        q_input2 = torch.quantize_per_tensor(input2, 0.5, zero_point, dtype_inputs)
        q_input3 = torch.quantize_per_tensor(input3, 0.6, zero_point, dtype_inputs)

        output_int8 = torch.ops.quantized.cat([q_input1, q_input2, q_input3], dim=1, scale=0.02, zero_point=0)

        q_input1_gpu = torch.quantize_per_tensor(input1_gpu, 0.4, zero_point, dtype_inputs)
        q_input2_gpu = torch.quantize_per_tensor(input2_gpu, 0.5, zero_point, dtype_inputs)
        q_input3_gpu = torch.quantize_per_tensor(input3_gpu, 0.6, zero_point, dtype_inputs)

        output_gpu_int8 = torch.ops.quantized.cat(
            [q_input1_gpu, q_input2_gpu, q_input3_gpu], dim=1, scale=0.02, zero_point=0)

        self.assertEqual(output_int8, output_gpu_int8)

    def test_cat_array_qint8(self, dtype=torch.float):
        zero_point = 0
        dtype_inputs = torch.qint8

        input1 = torch.randn(1, 1, 5, 5)
        input2 = torch.randn(1, 1, 5, 5)
        input3 = torch.randn(1, 1, 5, 5)

        input1_gpu = input1.to("xpu")
        input2_gpu = input2.to("xpu")
        input3_gpu = input3.to("xpu")

        q_input1 = torch.quantize_per_tensor(input1, 0.04, zero_point, dtype_inputs)
        q_input2 = torch.quantize_per_tensor(input2, 0.05, zero_point, dtype_inputs)
        q_input3 = torch.quantize_per_tensor(input3, 0.06, zero_point, dtype_inputs)

        output_int8 = torch.ops.quantized.cat([q_input1, q_input2, q_input3], dim=1, scale=0.2, zero_point=0)

        q_input1_gpu = torch.quantize_per_tensor(input1_gpu, 0.04, zero_point, dtype_inputs)
        q_input2_gpu = torch.quantize_per_tensor(input2_gpu, 0.05, zero_point, dtype_inputs)
        q_input3_gpu = torch.quantize_per_tensor(input3_gpu, 0.06, zero_point, dtype_inputs)

        output_gpu_int8 = torch.ops.quantized.cat(
            [q_input1_gpu, q_input2_gpu, q_input3_gpu], dim=1, scale=0.2, zero_point=0)

        self.assertEqual(output_int8, output_gpu_int8)

    def test_quint8_conv_cat_array_and_dequantize(self, dtype=torch.float):
        zero_point_in = 128
        zero_point_w = 0
        zero_point_out = 128
        dtype_inputs = torch.quint8
        dtype_weights = torch.qint8

        input1 = torch.randn(1, 1, 5, 5)
        input2 = torch.randn(1, 1, 5, 5)
        input3 = torch.randn(1, 1, 5, 5)
        input4 = torch.randn(1, 1, 5, 5)
        input5 = torch.randn(1, 1, 5, 5)
        input6 = torch.randn(1, 1, 5, 5)

        weight1 = torch.randn(3, 1, 3, 3)
        weight2 = torch.randn(3, 1, 3, 3)
        weight3 = torch.randn(3, 1, 3, 3)
        weight4 = torch.randn(3, 1, 3, 3)
        weight5 = torch.randn(3, 1, 3, 3)
        weight6 = torch.randn(3, 1, 3, 3)

        bias1 = torch.randn(3, dtype=torch.float)
        bias2 = torch.randn(3, dtype=torch.float)
        bias3 = torch.randn(3, dtype=torch.float)
        bias4 = torch.randn(3, dtype=torch.float)
        bias5 = torch.randn(3, dtype=torch.float)
        bias6 = torch.randn(3, dtype=torch.float)

        input1_gpu = input1.to("xpu")
        input2_gpu = input2.to("xpu")
        input3_gpu = input3.to("xpu")
        input4_gpu = input4.to("xpu")
        input5_gpu = input5.to("xpu")
        input6_gpu = input6.to("xpu")

        weight1_gpu = weight1.to("xpu")
        weight2_gpu = weight2.to("xpu")
        weight3_gpu = weight3.to("xpu")
        weight4_gpu = weight4.to("xpu")
        weight5_gpu = weight5.to("xpu")
        weight6_gpu = weight6.to("xpu")

        bias1_gpu = bias1.to("xpu")
        bias2_gpu = bias2.to("xpu")
        bias3_gpu = bias3.to("xpu")
        bias4_gpu = bias4.to("xpu")
        bias5_gpu = bias5.to("xpu")
        bias6_gpu = bias6.to("xpu")

        q_input1 = torch.quantize_per_tensor(input1, 0.04, zero_point_in, dtype_inputs)
        q_input2 = torch.quantize_per_tensor(input2, 0.05, zero_point_in, dtype_inputs)
        q_input3 = torch.quantize_per_tensor(input3, 0.06, zero_point_in, dtype_inputs)
        q_input4 = torch.quantize_per_tensor(input4, 0.06, zero_point_in, dtype_inputs)
        q_input5 = torch.quantize_per_tensor(input5, 0.06, zero_point_in, dtype_inputs)
        q_input6 = torch.quantize_per_tensor(input6, 0.06, zero_point_in, dtype_inputs)

        q_weight1 = torch.quantize_per_tensor(weight1, 0.04, zero_point_w, dtype_weights)
        q_weight2 = torch.quantize_per_tensor(weight2, 0.05, zero_point_w, dtype_weights)
        q_weight3 = torch.quantize_per_tensor(weight3, 0.06, zero_point_w, dtype_weights)
        q_weight4 = torch.quantize_per_tensor(weight4, 0.06, zero_point_w, dtype_weights)
        q_weight5 = torch.quantize_per_tensor(weight5, 0.06, zero_point_w, dtype_weights)
        q_weight6 = torch.quantize_per_tensor(weight6, 0.06, zero_point_w, dtype_weights)

        model = Fake_Q_Conv_Cat_Dequantize(1, 0.5, zero_point_in, [0.2, 0.1, 0.2, 0.1, 0.2, 0.1])
        output_cpu = model([q_input1, q_input2, q_input3, q_input4, q_input5, q_input6],
                           [q_weight1, q_weight2, q_weight3, q_weight4, q_weight5, q_weight6],
                           [bias1, bias2, bias3, bias4, bias5, bias6])

        q_input1_gpu = torch.quantize_per_tensor(input1_gpu, 0.04, zero_point_in, dtype_inputs)
        q_input2_gpu = torch.quantize_per_tensor(input2_gpu, 0.05, zero_point_in, dtype_inputs)
        q_input3_gpu = torch.quantize_per_tensor(input3_gpu, 0.06, zero_point_in, dtype_inputs)
        q_input4_gpu = torch.quantize_per_tensor(input4_gpu, 0.06, zero_point_in, dtype_inputs)
        q_input5_gpu = torch.quantize_per_tensor(input5_gpu, 0.06, zero_point_in, dtype_inputs)
        q_input6_gpu = torch.quantize_per_tensor(input6_gpu, 0.06, zero_point_in, dtype_inputs)

        q_weight1_gpu = torch.quantize_per_tensor(weight1_gpu, 0.04, zero_point_w, dtype_weights)
        q_weight2_gpu = torch.quantize_per_tensor(weight2_gpu, 0.05, zero_point_w, dtype_weights)
        q_weight3_gpu = torch.quantize_per_tensor(weight3_gpu, 0.06, zero_point_w, dtype_weights)
        q_weight4_gpu = torch.quantize_per_tensor(weight4_gpu, 0.06, zero_point_w, dtype_weights)
        q_weight5_gpu = torch.quantize_per_tensor(weight5_gpu, 0.06, zero_point_w, dtype_weights)
        q_weight6_gpu = torch.quantize_per_tensor(weight6_gpu, 0.06, zero_point_w, dtype_weights)

        intput_gpu = (q_input1_gpu, q_input2_gpu, q_input3_gpu, q_input4_gpu, q_input5_gpu, q_input6_gpu)
        weight_gpu = (q_weight1_gpu, q_weight2_gpu, q_weight3_gpu, q_weight4_gpu, q_weight5_gpu, q_weight6_gpu)
        bias_gpu = (bias1_gpu, bias2_gpu, bias3_gpu, bias4_gpu, bias5_gpu, bias6_gpu)
        xpu_model = Q_Conv_Cat_Dequantize(1, 0.5, zero_point_out, [0.2, 0.1, 0.2, 0.1, 0.2, 0.1], False)
        xpu_model.to("xpu")
        modelJit = torch.jit.trace(xpu_model, (intput_gpu, weight_gpu, bias_gpu), check_trace=False)
        with torch.no_grad():
            for i in range(2):
                output_gpu = modelJit(intput_gpu, weight_gpu, bias_gpu)
        self.assertEqual(output_cpu, output_gpu.cpu())

    def test_quint8_conv_relu_cat_array_and_dequantize(self, dtype=torch.float):
        zero_point_in = 128
        zero_point_w = 0
        zero_point_out = 128
        dtype_inputs = torch.quint8
        dtype_weights = torch.qint8

        input1 = torch.randn(1, 1, 5, 5)
        input2 = torch.randn(1, 1, 5, 5)
        input3 = torch.randn(1, 1, 5, 5)
        input4 = torch.randn(1, 1, 5, 5)
        input5 = torch.randn(1, 1, 5, 5)
        input6 = torch.randn(1, 1, 5, 5)

        weight1 = torch.randn(3, 1, 3, 3)
        weight2 = torch.randn(3, 1, 3, 3)
        weight3 = torch.randn(3, 1, 3, 3)
        weight4 = torch.randn(3, 1, 3, 3)
        weight5 = torch.randn(3, 1, 3, 3)
        weight6 = torch.randn(3, 1, 3, 3)

        bias1 = torch.randn(3, dtype=torch.float)
        bias2 = torch.randn(3, dtype=torch.float)
        bias3 = torch.randn(3, dtype=torch.float)
        bias4 = torch.randn(3, dtype=torch.float)
        bias5 = torch.randn(3, dtype=torch.float)
        bias6 = torch.randn(3, dtype=torch.float)

        input1_gpu = input1.to("xpu")
        input2_gpu = input2.to("xpu")
        input3_gpu = input3.to("xpu")
        input4_gpu = input4.to("xpu")
        input5_gpu = input5.to("xpu")
        input6_gpu = input6.to("xpu")

        weight1_gpu = weight1.to("xpu")
        weight2_gpu = weight2.to("xpu")
        weight3_gpu = weight3.to("xpu")
        weight4_gpu = weight4.to("xpu")
        weight5_gpu = weight5.to("xpu")
        weight6_gpu = weight6.to("xpu")

        bias1_gpu = bias1.to("xpu")
        bias2_gpu = bias2.to("xpu")
        bias3_gpu = bias3.to("xpu")
        bias4_gpu = bias4.to("xpu")
        bias5_gpu = bias5.to("xpu")
        bias6_gpu = bias6.to("xpu")

        q_input1 = torch.quantize_per_tensor(input1, 0.04, zero_point_in, dtype_inputs)
        q_input2 = torch.quantize_per_tensor(input2, 0.05, zero_point_in, dtype_inputs)
        q_input3 = torch.quantize_per_tensor(input3, 0.06, zero_point_in, dtype_inputs)
        q_input4 = torch.quantize_per_tensor(input4, 0.06, zero_point_in, dtype_inputs)
        q_input5 = torch.quantize_per_tensor(input5, 0.06, zero_point_in, dtype_inputs)
        q_input6 = torch.quantize_per_tensor(input6, 0.06, zero_point_in, dtype_inputs)

        q_weight1 = torch.quantize_per_tensor(weight1, 0.04, zero_point_w, dtype_weights)
        q_weight2 = torch.quantize_per_tensor(weight2, 0.05, zero_point_w, dtype_weights)
        q_weight3 = torch.quantize_per_tensor(weight3, 0.06, zero_point_w, dtype_weights)
        q_weight4 = torch.quantize_per_tensor(weight4, 0.06, zero_point_w, dtype_weights)
        q_weight5 = torch.quantize_per_tensor(weight5, 0.06, zero_point_w, dtype_weights)
        q_weight6 = torch.quantize_per_tensor(weight6, 0.06, zero_point_w, dtype_weights)

        model = Fake_Q_Conv_Relu_Cat_Dequantize(1, 0.5, zero_point_in, [0.2, 0.1, 0.2, 0.1, 0.2, 0.1])
        output_cpu = model([q_input1, q_input2, q_input3, q_input4, q_input5, q_input6],
                           [q_weight1, q_weight2, q_weight3, q_weight4, q_weight5, q_weight6],
                           [bias1, bias2, bias3, bias4, bias5, bias6])

        q_input1_gpu = torch.quantize_per_tensor(input1_gpu, 0.04, zero_point_in, dtype_inputs)
        q_input2_gpu = torch.quantize_per_tensor(input2_gpu, 0.05, zero_point_in, dtype_inputs)
        q_input3_gpu = torch.quantize_per_tensor(input3_gpu, 0.06, zero_point_in, dtype_inputs)
        q_input4_gpu = torch.quantize_per_tensor(input4_gpu, 0.06, zero_point_in, dtype_inputs)
        q_input5_gpu = torch.quantize_per_tensor(input5_gpu, 0.06, zero_point_in, dtype_inputs)
        q_input6_gpu = torch.quantize_per_tensor(input6_gpu, 0.06, zero_point_in, dtype_inputs)

        q_weight1_gpu = torch.quantize_per_tensor(weight1_gpu, 0.04, zero_point_w, dtype_weights)
        q_weight2_gpu = torch.quantize_per_tensor(weight2_gpu, 0.05, zero_point_w, dtype_weights)
        q_weight3_gpu = torch.quantize_per_tensor(weight3_gpu, 0.06, zero_point_w, dtype_weights)
        q_weight4_gpu = torch.quantize_per_tensor(weight4_gpu, 0.06, zero_point_w, dtype_weights)
        q_weight5_gpu = torch.quantize_per_tensor(weight5_gpu, 0.06, zero_point_w, dtype_weights)
        q_weight6_gpu = torch.quantize_per_tensor(weight6_gpu, 0.06, zero_point_w, dtype_weights)

        intput_gpu = (q_input1_gpu, q_input2_gpu, q_input3_gpu, q_input4_gpu, q_input5_gpu, q_input6_gpu)
        weight_gpu = (q_weight1_gpu, q_weight2_gpu, q_weight3_gpu, q_weight4_gpu, q_weight5_gpu, q_weight6_gpu)
        bias_gpu = (bias1_gpu, bias2_gpu, bias3_gpu, bias4_gpu, bias5_gpu, bias6_gpu)
        xpu_model = Q_Conv_Relu_Cat_Dequantize(1, 1.0, zero_point_out, [0.4, 0.2, 0.4, 0.2, 0.4, 0.2], False)
        xpu_model.to("xpu")
        modelJit = torch.jit.trace(xpu_model, (intput_gpu, weight_gpu, bias_gpu), check_trace=False)
        with torch.no_grad():
            for i in range(2):
                output_gpu = modelJit(intput_gpu, weight_gpu, bias_gpu)
        self.assertEqual(output_cpu, output_gpu.cpu())

    def test_qint8_cat_array_and_dequantize(self, dtype=torch.float):
        zero_point = 0
        dtype_inputs = torch.qint8
        model = Q_Cat_Dequantize(1, 0.5, 0)
        model_fake = Fake_Q_Cat_Dequantize(1, 0.5, 0)

        input1 = torch.randn(1, 1, 50, 500)
        input2 = torch.randn(1, 1, 50, 500)
        input3 = torch.randn(1, 1, 50, 500)

        input1_gpu = input1.to("xpu")
        input2_gpu = input2.to("xpu")
        input3_gpu = input3.to("xpu")

        q_input1 = torch.quantize_per_tensor(input1, 0.04, zero_point, dtype_inputs)
        q_input2 = torch.quantize_per_tensor(input2, 0.05, zero_point, dtype_inputs)
        q_input3 = torch.quantize_per_tensor(input3, 0.06, zero_point, dtype_inputs)

        output_cpu = model_fake([q_input1, q_input2, q_input3])
        q_input1_gpu = torch.quantize_per_tensor(input1_gpu, 0.04, zero_point, dtype_inputs)
        q_input2_gpu = torch.quantize_per_tensor(input2_gpu, 0.05, zero_point, dtype_inputs)
        q_input3_gpu = torch.quantize_per_tensor(input3_gpu, 0.06, zero_point, dtype_inputs)

        intput_gpu = [q_input1_gpu, q_input2_gpu, q_input3_gpu]
        model.to("xpu")
        modelJit = torch.jit.trace(model, (intput_gpu,), check_trace=False)
        with torch.no_grad():
            for i in range(2):
                output_gpu = modelJit([q_input1_gpu, q_input2_gpu, q_input3_gpu])
        self.assertEqual(output_cpu, output_gpu.cpu())

    def test_quint8_cat_array_and_dequantize(self, dtype=torch.float):
        zero_point = 0
        dtype_inputs = torch.quint8
        model = Q_Cat_Dequantize(1, 0.5, 0)
        model_fake = Fake_Q_Cat_Dequantize(1, 0.5, 0)

        input1 = torch.randn(1, 1, 50, 500)
        input2 = torch.randn(1, 1, 50, 500)
        input3 = torch.randn(1, 1, 50, 500)

        input1_gpu = input1.to("xpu")
        input2_gpu = input2.to("xpu")
        input3_gpu = input3.to("xpu")

        q_input1 = torch.quantize_per_tensor(input1, 0.04, zero_point, dtype_inputs)
        q_input2 = torch.quantize_per_tensor(input2, 0.05, zero_point, dtype_inputs)
        q_input3 = torch.quantize_per_tensor(input3, 0.06, zero_point, dtype_inputs)

        output_cpu = model_fake([q_input1, q_input2, q_input3])
        q_input1_gpu = torch.quantize_per_tensor(input1_gpu, 0.04, zero_point, dtype_inputs)
        q_input2_gpu = torch.quantize_per_tensor(input2_gpu, 0.05, zero_point, dtype_inputs)
        q_input3_gpu = torch.quantize_per_tensor(input3_gpu, 0.06, zero_point, dtype_inputs)

        intput_gpu = [q_input1_gpu, q_input2_gpu, q_input3_gpu]
        model.to("xpu")
        modelJit = torch.jit.trace(model, (intput_gpu,), check_trace=False)
        with torch.no_grad():
            for i in range(2):
                output_gpu = modelJit([q_input1_gpu, q_input2_gpu, q_input3_gpu])
        self.assertEqual(output_cpu, output_gpu.cpu())
