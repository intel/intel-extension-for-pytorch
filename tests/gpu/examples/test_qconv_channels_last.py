import torch
from torch.nn.modules.utils import _pair
import pytest
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        "fbgemm" not in torch.backends.quantized.supported_engines,
        reason="No qengine found. USE_FBGEMM=1 is needed for building pytorch",
    )
    def test_qconv_simple_channels_last(self, dtype=torch.float):
        dtype_inputs = torch.quint8
        dtype_filters = torch.qint8
        zp_vec = [128, 2, 0]
        for with_relu in [True, False]:
            qconf_fn = torch.ops.quantized.conv2d
            if with_relu:
                qconv_fn = torch.ops.quantized.conv2d_relu
            for scale_in in [1.2, 1.6]:
                for zero_point_in in zp_vec:  # torch u8, random zp, 0
                    scale_weight = 0.5
                    scale_out = 0.5
                    zero_point_out = 2

                    inputs = torch.randn(1, 2, 5, 5)
                    filters = torch.randn(4, 2, 3, 3)
                    bias = torch.randn(4)

                    # Here use 128 as zp to preserving negative value in FP32 tensor
                    q_inputs = torch.quantize_per_tensor(
                        inputs, scale_in, zero_point_in, dtype_inputs
                    )
                    q_filters = torch.quantize_per_tensor(
                        filters, scale_weight, 0, dtype_filters
                    )

                    packed_params = torch.ops.quantized.conv2d_prepack(
                        q_filters, bias, _pair(1), _pair(0), _pair(1), 1
                    )
                    # We utilize the torch's asymmetric design here.
                    # Specifically, we let out of conv_relu has same zp and scale as oneDNN requres.
                    output_int8 = qconv_fn(
                        q_inputs, packed_params, scale_out, zero_point_out
                    )

                    inputs = inputs.to(memory_format=torch.channels_last)
                    inputs_gpu = inputs.to("xpu")
                    filters_gpu = filters.to("xpu")
                    bias_gpu = bias.to("xpu")

                    q_inputs_gpu = torch.quantize_per_tensor(
                        inputs_gpu, scale_in, zero_point_in, dtype_inputs
                    )
                    q_filters_gpu = torch.quantize_per_tensor(
                        filters_gpu, scale_weight, 0, dtype_filters
                    )

                    packed_params_gpu = torch.ops.quantized.conv2d_prepack(
                        q_filters_gpu, bias_gpu, _pair(1), _pair(0), _pair(1), 1
                    )
                    # We want every usage is just like pytorch.
                    output_gpu_int8 = qconv_fn(
                        q_inputs_gpu, packed_params_gpu, scale_out, zero_point_out
                    )

                    cpu_result = torch.dequantize(output_int8)
                    gpu_result = torch.dequantize(output_gpu_int8).cpu().contiguous()

                    print(cpu_result)
                    print(gpu_result)

                    self.assertEqual(cpu_result, gpu_result)

    @pytest.mark.skipif(
        "fbgemm" not in torch.backends.quantized.supported_engines,
        reason="No qengine found. USE_FBGEMM=1 is needed for building pytorch",
    )
    def test_qconv_simple_channels_last_3d(self, dtype=torch.float):
        dtype_inputs = torch.quint8
        dtype_filters = torch.qint8
        zp_vec = [128, 2, 0]
        for with_relu in [True, False]:
            qconf_fn = torch.ops.quantized.conv3d
            if with_relu:
                qconv_fn = torch.ops.quantized.conv3d_relu
            for scale_in in [1.2, 1.6]:
                for zero_point_in in zp_vec:  # torch u8, random zp, 0
                    scale_weight = 0.5
                    scale_out = 4.0
                    zero_point_out = 2

                    inputs = torch.randn(1, 2, 5, 5, 5)
                    filters = torch.randn(4, 2, 3, 3, 3)
                    bias = torch.randn(4)

                    # Here use 128 as zp to preserving negative value in FP32 tensor
                    q_inputs = torch.quantize_per_tensor(
                        inputs, scale_in, zero_point_in, dtype_inputs
                    )
                    q_filters = torch.quantize_per_tensor(
                        filters, scale_weight, 0, dtype_filters
                    )

                    stride = (1, 1, 1)
                    padding = (0, 0, 0)
                    dilation = (1, 1, 1)
                    packed_params = torch.ops.quantized.conv3d_prepack(
                        q_filters, bias, stride, padding, dilation, 1
                    )
                    # We utilize the torch's asymmetric design here.
                    # Specifically, we let out of conv_relu has same zp and scale as oneDNN requres.
                    output_int8 = qconv_fn(
                        q_inputs, packed_params, scale_out, zero_point_out
                    )

                    inputs = inputs.to(memory_format=torch.channels_last_3d)
                    inputs_gpu = inputs.to("xpu")
                    filters_gpu = filters.to("xpu")
                    bias_gpu = bias.to("xpu")
                    q_inputs_gpu = torch.quantize_per_tensor(
                        inputs_gpu, scale_in, zero_point_in, dtype_inputs
                    )
                    q_filters_gpu = torch.quantize_per_tensor(
                        filters_gpu, scale_weight, 0, dtype_filters
                    )

                    packed_params_gpu = torch.ops.quantized.conv3d_prepack(
                        q_filters_gpu, bias_gpu, stride, padding, dilation, 1
                    )
                    # We wand every usage is just like pytorch.
                    output_gpu_int8 = qconv_fn(
                        q_inputs_gpu, packed_params_gpu, scale_out, zero_point_out
                    )

                    cpu_result = torch.dequantize(output_int8)
                    gpu_result = torch.dequantize(output_gpu_int8).cpu().contiguous()

                    print(cpu_result)
                    print(gpu_result)

                    self.assertEqual(cpu_result, gpu_result)
