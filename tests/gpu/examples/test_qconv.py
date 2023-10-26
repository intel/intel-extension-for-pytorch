import torch
from torch.nn.modules.utils import _pair
import pytest
import numpy as np
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        "fbgemm" not in torch.backends.quantized.supported_engines,
        reason="No engine found. USE_FBGEMM=1 is needed for building pytorch",
    )
    def test_qconv2d(self, dtype=torch.float):
        print(
            "Please open FBGEMM(Pytorch CPU INT8 default engihe) when build Pytorch,"
            "which ill be referenced by GPU. Or you may meet runtime error like"
            "'Didn't find engine for operation quantized::conv3d_prepack'."
        )
        dtype_inputs = torch.quint8
        dtype_filters = torch.qint8
        zp_vec = [128, 2, 0]
        for with_relu in [False, True]:
            for scale_in in [1.2, 1.6]:
                for zero_point_in in zp_vec:  # torch u8, random zp, 0
                    scale_weight = 0.2
                    inputs = torch.randn(1, 2, 5, 5)
                    filters = torch.randn(4, 2, 3, 3)
                    bias = torch.randn(4)

                    scale_out = 4.2
                    zp_out = np.random.randint(0, 255)
                    print(
                        f"with_relu:{with_relu}, scale:{scale_in}, zero_point:{zero_point_in}, zp_out:{zp_out}"
                    )

                    if with_relu:
                        qconv_fn = torch.ops.quantized.conv2d_relu
                    else:
                        qconv_fn = torch.ops.quantized.conv2d

                    q_inputs = torch.quantize_per_tensor(
                        inputs, scale_in, zero_point_in, dtype_inputs
                    )
                    q_filters = torch.quantize_per_tensor(
                        filters, scale_weight, 0, dtype_filters
                    )

                    packed_params = torch.ops.quantized.conv2d_prepack(
                        q_filters, bias, _pair(1), _pair(0), _pair(1), 1
                    )
                    output_int8 = qconv_fn(q_inputs, packed_params, scale_out, zp_out)

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
                    output_gpu_int8 = qconv_fn(
                        q_inputs_gpu, packed_params_gpu, scale_out, zp_out
                    )
                    cpu_result = torch.dequantize(output_int8)
                    gpu_result = torch.dequantize(output_gpu_int8)
                    self.assertEqual(cpu_result, gpu_result)

    @pytest.mark.skipif(
        "fbgemm" not in torch.backends.quantized.supported_engines,
        reason="No engine found. USE_FBGEMM=1 is needed for building pytorch",
    )
    def test_qconv3d(self, dtype=torch.float):
        print(
            "Please open FBGEMM(Pytorch CPU INT8 default engihe) when build Pytorch,"
            "which ill be referenced by GPU. Or you may meet runtime error like"
            "'Didn't find engine for operation quantized::conv3d_prepack'."
        )

        dtype_inputs = torch.quint8
        dtype_filters = torch.qint8
        zp_vec = [128, 2, 0]
        for with_relu in [True, False]:
            for scale_in in [1.2, 1.6]:
                for zero_point_in in zp_vec:  # torch u8, random zp, 0
                    inputs = torch.randn(1, 2, 5, 5, 5)
                    filters = torch.randn(4, 2, 3, 3, 3)
                    bias = torch.randn(4)

                    scale_weight = 0.5
                    zp_out = np.random.randint(0, 255)
                    scale_out = 0.35
                    print(
                        f"with_relu:{with_relu}, scale_in:{scale_in}, \
                        zero_point_in:{zero_point_in}, scale_out:{scale_out}, zp_out:{zp_out}"
                    )

                    if with_relu:
                        qconv_fn = torch.ops.quantized.conv3d_relu
                    else:
                        qconv_fn = torch.ops.quantized.conv3d

                    q_inputs = torch.quantize_per_tensor(
                        inputs, scale_in, zero_point_in, dtype_inputs
                    )
                    q_filters = torch.quantize_per_tensor(
                        filters, scale_weight, 0, dtype_filters
                    )

                    packed_params = torch.ops.quantized.conv3d_prepack(
                        q_filters, bias, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1
                    )
                    output_int8 = qconv_fn(q_inputs, packed_params, scale_out, zp_out)

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
                        q_filters_gpu, bias_gpu, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1
                    )
                    output_gpu_int8 = qconv_fn(
                        q_inputs_gpu, packed_params_gpu, scale_out, zp_out
                    )

                    cpu_result = torch.dequantize(output_int8)
                    gpu_result = torch.dequantize(output_gpu_int8)

                    self.assertEqual(cpu_result, gpu_result)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_qconv_per_channel(self, dtype=torch.float):
        print(
            "Please open FBGEMM (PyTorch CPU INT8 default engine ) when build PyTorch, "
            "which will be referenced by GPU. Or you may meet runtime error like "
            "'Didn't find engine for operation quantized::conv2d_prepack'."
        )
        dtype_inputs = torch.quint8
        dtype_filters = torch.qint8

        zp_vec = [128, 2, 0]
        for with_relu in [True, False]:
            for scale_in in [1.2, 1.6]:
                for zero_point_in in zp_vec:  # torch u8, random zp, 0
                    scale_weight = 0.5
                    inputs = torch.randn(1, 2, 5, 5)
                    filters = torch.randn(4, 2, 3, 3)
                    filter_scale = torch.empty((4), dtype=torch.float).fill_(0.5)
                    filter_zero_point = torch.empty((4), dtype=torch.int32).fill_(0)
                    bias = torch.randn(4)
                    if with_relu:
                        qconv_fn = torch.ops.quantized.conv2d_relu
                    else:
                        qconv_fn = torch.ops.quantized.conv2d

                    zp_out = np.random.randint(0, 255)
                    scale_out = 0.35

                    q_inputs = torch.quantize_per_tensor(
                        inputs, scale_in, zero_point_in, dtype_inputs
                    )
                    q_filters = torch.quantize_per_channel(
                        filters, filter_scale, filter_zero_point, 0, dtype_filters
                    )

                    packed_params = torch.ops.quantized.conv2d_prepack(
                        q_filters, bias, _pair(1), _pair(0), _pair(1), 1
                    )
                    output_int8 = qconv_fn(
                        q_inputs,
                        packed_params,
                        scale_out,
                        zp_out,
                    )

                    inputs_gpu = inputs.to("xpu")
                    filters_gpu = filters.to("xpu")
                    bias_gpu = bias.to("xpu")
                    filter_scale_gpu = filter_scale.to("xpu")
                    filter_zero_point_gpu = filter_zero_point.to("xpu")

                    q_inputs_gpu = torch.quantize_per_tensor(
                        inputs_gpu, scale_in, zero_point_in, dtype_inputs
                    )
                    q_filters_gpu = torch.quantize_per_channel(
                        filters_gpu,
                        filter_scale_gpu,
                        filter_zero_point_gpu,
                        0,
                        dtype_filters,
                    )
                    packed_params_gpu = torch.ops.quantized.conv2d_prepack(
                        q_filters_gpu, bias_gpu, _pair(1), _pair(0), _pair(1), 1
                    )
                    output_gpu_int8 = qconv_fn(
                        q_inputs_gpu, packed_params_gpu, scale_out, zp_out
                    )
                    cpu_result = torch.dequantize(output_int8)
                    gpu_result = torch.dequantize(output_gpu_int8)
                    self.assertEqual(cpu_result, gpu_result)
