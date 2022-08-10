import torch
from torch.nn.modules.utils import _pair
import pytest
import intel_extension_for_pytorch # noqa
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    @pytest.mark.skipif('fbgemm' not in torch.backends.quantized.supported_engines,
                        reason="No engine found. USE_FBGEMM=1 is needed for building pytorch")
    def test_pure_qconv2d(self, dtype=torch.float):

        print("Please open FBGEMM(Pytorch CPU INT8 default engihe) when build Pytorch,"
              "which ill be referenced by GPU. Or you may meet runtime error like"
              "'Didn't find engine for operation quantized::conv3d_prepack'.")

        u8_zp = 128  # bg info: CPU use 128 as zp for u8 qtensor, while 0 is in oneDNN
        s8_zp = 0  # Both cpu and xpu use 0 as zp respect to s8
        dtype_inputs = torch.quint8
        dtype_filters = torch.qint8

        scale_in = 0.4
        scale_weight = 0.5
        scale_out = 4.0

        inputs = torch.randn(1, 2, 5, 5)
        filters = torch.randn(4, 2, 3, 3)
        bias = torch.randn(4)

        q_inputs = torch.quantize_per_tensor(inputs, scale_in, u8_zp, dtype_inputs)  # For cpu, u8 input has zp 128
        q_filters = torch.quantize_per_tensor(filters, scale_weight, s8_zp, dtype_filters)  # s8 weight has zp 0

        packed_params = torch.ops.quantized.conv2d_prepack(q_filters, bias, _pair(1), _pair(0), _pair(1), 1)
        output_int8 = torch.ops.quantized.conv2d(q_inputs, packed_params, _pair(1), _pair(0), _pair(1), 1,
                                                 scale_out, u8_zp)
        # CPU conv has u8 output, use 128 as zp

        inputs_gpu = inputs.to("xpu")
        filters_gpu = filters.to("xpu")
        bias_gpu = bias.to("xpu")

        q_inputs_gpu = torch.quantize_per_tensor(inputs_gpu, scale_in, u8_zp, dtype_inputs)
        # At xpu side, quantize_per_tensor also accpet 128 zp for u8 qtensor, we reassign zp to 0 in cpp backend.
        q_filters_gpu = torch.quantize_per_tensor(filters_gpu, scale_weight, s8_zp, dtype_filters)  # s8 weight, zp use 0
        packed_params_gpu = torch.ops.quantized.conv2d_prepack(q_filters_gpu, bias_gpu, _pair(1), _pair(0), _pair(1), 1)
        output_gpu_int8 = torch.ops.quantized.conv2d(q_inputs_gpu, packed_params_gpu, scale_out, s8_zp)  # s8 output, use 0 as zp

        cpu_result = torch.dequantize(output_int8)
        gpu_result = torch.dequantize(output_gpu_int8)
        self.assertEqual(cpu_result, gpu_result)

    @pytest.mark.skipif('fbgemm' not in torch.backends.quantized.supported_engines,
                        reason="No engine found. USE_FBGEMM=1 is needed for building pytorch")
    def test_pure_qconv3d(self, dtype=torch.float):

        print("Please open FBGEMM(Pytorch CPU INT8 default engihe) when build Pytorch,"
              "which ill be referenced by GPU. Or you may meet runtime error like"
              "'Didn't find engine for operation quantized::conv3d_prepack'.")

        # TODO: check 2d usage
        u8_zp = 128
        s8_zp = 0
        dtype_inputs = torch.quint8
        dtype_filters = torch.qint8

        scale_in = 0.4
        scale_weight = 0.5
        scale_out = 4.0

        inputs = torch.randn(1, 2, 5, 5, 5)
        filters = torch.randn(4, 2, 3, 3, 3)
        bias = torch.randn(4)

        q_inputs = torch.quantize_per_tensor(inputs, scale_in, u8_zp, dtype_inputs)
        q_filters = torch.quantize_per_tensor(filters, scale_weight, s8_zp, dtype_filters)

        packed_params = torch.ops.quantized.conv3d_prepack(q_filters, bias, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1)
        output_int8 = torch.ops.quantized.conv3d(q_inputs, packed_params, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1, scale_out, u8_zp)

        inputs_gpu = inputs.to("xpu")
        filters_gpu = filters.to("xpu")
        bias_gpu = bias.to("xpu")

        q_inputs_gpu = torch.quantize_per_tensor(inputs_gpu, scale_in, u8_zp, dtype_inputs)
        q_filters_gpu = torch.quantize_per_tensor(filters_gpu, scale_weight, s8_zp, dtype_filters)

        packed_params_gpu = torch.ops.quantized.conv3d_prepack(q_filters_gpu, bias_gpu, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1)
        output_gpu_int8 = torch.ops.quantized.conv3d(q_inputs_gpu, packed_params_gpu, scale_out, s8_zp)


        cpu_result = torch.dequantize(output_int8)
        gpu_result = torch.dequantize(output_gpu_int8)

        self.assertEqual(cpu_result, gpu_result)


    @pytest.mark.skipif('fbgemm' not in torch.backends.quantized.supported_engines,
                        reason="No engine found. USE_FBGEMM=1 is needed for building pytorch")
    def test_qconv3d(self, dtype=torch.float):

        print("Please open FBGEMM(Pytorch CPU INT8 default engihe) when build Pytorch,"
              "which ill be referenced by GPU. Or you may meet runtime error like"
              "'Didn't find engine for operation quantized::conv3d_prepack'.")

        # TODO: check 2d usage
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

        packed_params = torch.ops.quantized.conv3d_prepack(q_filters, bias, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1)
        output_int8 = torch.ops.quantized.conv3d_relu(
            q_inputs, packed_params, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1, scale_out, zero_point)
        output_int8_2 = torch.ops.quantized.conv3d_relu(
            q_inputs, packed_params, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1, scale_out_2, zero_point)
        output_int8_3 = torch.ops.quantized.conv3d_relu(
            q_inputs, packed_params, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1, scale_out_2, zero_point)

        inputs_gpu = inputs.to("xpu")
        filters_gpu = filters.to("xpu")
        bias_gpu = bias.to("xpu")

        q_inputs_gpu = torch.quantize_per_tensor(inputs_gpu, scale_in, zero_point, dtype_inputs)
        q_filters_gpu = torch.quantize_per_tensor(filters_gpu, scale_weight, zero_point, dtype_filters)

        packed_params_gpu = torch.ops.quantized.conv3d_prepack(
            q_filters_gpu, bias_gpu, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1)
        output_gpu_int8 = torch.ops.quantized.conv3d_relu(q_inputs_gpu, packed_params_gpu, scale_out, zero_point)
        output_gpu_int8_2 = torch.ops.quantized.conv3d_relu(q_inputs_gpu, packed_params_gpu, scale_out_2, zero_point)
        output_gpu_int8_3 = torch.ops.quantized.conv3d_relu(q_inputs_gpu, packed_params_gpu, scale_out_2, zero_point)

        cpu_result = torch.dequantize(output_int8)
        gpu_result = torch.dequantize(output_gpu_int8)

        cpu_result_2 = torch.dequantize(output_int8_2)
        gpu_result_2 = torch.dequantize(output_gpu_int8_2)

        cpu_result_3 = torch.dequantize(output_int8_3)
        gpu_result_3 = torch.dequantize(output_gpu_int8_3)

        self.assertEqual(cpu_result, gpu_result)
        self.assertEqual(cpu_result_2, gpu_result_2)
        self.assertEqual(cpu_result_3, gpu_result_3)

    @pytest.mark.skipif('fbgemm' not in torch.backends.quantized.supported_engines,
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

        inputs = torch.randn(1, 2, 5, 5)
        filters = torch.randn(4, 2, 3, 3)
        bias = torch.randn(4)

        q_inputs = torch.quantize_per_tensor(inputs, scale_in, zero_point, dtype_inputs)
        q_filters = torch.quantize_per_tensor(filters, scale_weight, zero_point, dtype_filters)

        packed_params = torch.ops.quantized.conv2d_prepack(q_filters, bias, _pair(1), _pair(0), _pair(1), 1)
        output_int8 = torch.ops.quantized.conv2d_relu(q_inputs, packed_params, _pair(1),
                                                      _pair(0), _pair(1), 1, scale_out, zero_point)
        output_int8_2 = torch.ops.quantized.conv2d_relu(q_inputs, packed_params, _pair(1),
                                                        _pair(0), _pair(1), 1, scale_out_2, zero_point)
        output_int8_3 = torch.ops.quantized.conv2d_relu(q_inputs, packed_params, _pair(1),
                                                        _pair(0), _pair(1), 1, scale_out_2, zero_point)

        inputs_gpu = inputs.to("xpu")
        filters_gpu = filters.to("xpu")
        bias_gpu = bias.to("xpu")

        q_inputs_gpu = torch.quantize_per_tensor(inputs_gpu, scale_in, zero_point, dtype_inputs)
        q_filters_gpu = torch.quantize_per_tensor(filters_gpu, scale_weight, zero_point, dtype_filters)
        packed_params_gpu = torch.ops.quantized.conv2d_prepack(q_filters_gpu, bias_gpu, _pair(1), _pair(0), _pair(1), 1)
        output_gpu_int8 = torch.ops.quantized.conv2d_relu(q_inputs_gpu, packed_params_gpu, scale_out, zero_point)
        output_gpu_int8_2 = torch.ops.quantized.conv2d_relu(q_inputs_gpu, packed_params_gpu, scale_out_2, zero_point)
        output_gpu_int8_3 = torch.ops.quantized.conv2d_relu(q_inputs_gpu, packed_params_gpu, scale_out_2, zero_point)

        cpu_result = torch.dequantize(output_int8)
        gpu_result = torch.dequantize(output_gpu_int8)

        cpu_result_2 = torch.dequantize(output_int8_2)
        gpu_result_2 = torch.dequantize(output_gpu_int8_2)

        cpu_result_3 = torch.dequantize(output_int8_3)
        gpu_result_3 = torch.dequantize(output_gpu_int8_3)

        self.assertEqual(cpu_result, gpu_result)
        self.assertEqual(cpu_result_2, gpu_result_2)
        self.assertEqual(cpu_result_3, gpu_result_3)
