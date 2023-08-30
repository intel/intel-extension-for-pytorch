import torch
import pytest
import intel_extension_for_pytorch as ipex  # noqa
from torch.testing._internal.common_utils import TestCase
import platform


def fake_minmax_sc(x):
    act_min, act_max = torch.aminmax(torch.abs(x))
    scale_input = 32 * act_max / 255
    return scale_input


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        "fbgemm" not in torch.backends.quantized.supported_engines,
        reason="No qengine found. USE_FBGEMM=1 is needed for building pytorch",
    )
    def test_qdeconv3d_cpuref(self, dtype=torch.float):
        print(
            "Please open FBGEMM (PyTorch CPU INT8 default engine ) when build PyTorch, "
            "which will be referenced by GPU. Or you may meet runtime error like "
            "'Didn't find engine for operation quantized::conv2d_prepack'."
        )
        with torch.xpu.onednn_verbose(0):
            zero_point = 0

            dtype_inputs = torch.quint8
            dtype_filters = torch.qint8

            output_channels = 1
            bias = None
            X_scale = 1.2
            X_zero_points = 0
            W_scale = 0.2
            Y_scale = 4.2
            Y_zero_point = 0
            X_zero_point = 0 if platform.system() == 'Windows' else 128

            (X_value_min, X_valu_max) = (32, 64)
            X_init = torch.randint(X_value_min, X_valu_max, (1, 1, 5, 5, 5))
            # Actually, this is a dequant formulation, motivation for this design is to make
            # true int8 tensor is at smalle range, aka (0, 5) in current case. This would
            # avoid overflow in int8 kernel computation.
            X = X_scale * (X_init - X_zero_point).float()

            # Deconv3d has more elemeents to calculate in one kernl than Deconv2d,
            # deconv3d is much more prone to cause overflow. We restrict qweight
            # only has [-1, 0, 1] to reduce the influece of overflow.
            (W_value_min, W_value_max) = (-1, 1)
            W_scale = W_scale * output_channels
            W_init = torch.randint(W_value_min, W_value_max, (1, 1, 2, 2, 2))
            W = (W_scale * W_init).float()

            q_inputs = torch.quantize_per_tensor(X, X_scale, X_zero_point, dtype_inputs)
            q_filters = torch.quantize_per_tensor(W, W_scale, 0, dtype_filters)

            packed_params = torch.ops.quantized.conv_transpose3d_prepack(
                q_filters, bias, (1, 1, 1), (0, 0, 0), (0, 0, 0), (1, 1, 1), 1
            )
            # We intend to let qconv_relu output has the ame quant scheme as oneDNN u8, aka scale/2 + 0
            output_int8 = torch.ops.quantized.conv_transpose3d(
                q_inputs, packed_params, Y_scale, Y_zero_point
            )
            print("cpu output result:", output_int8)
            print("cpu output result int_repr():", output_int8.int_repr())

            X_gpu = X.to("xpu")
            W_gpu = W.to("xpu")
            if bias is not None:
                bias_gpu = bias.to("xpu")
            else:
                bias_gpu = None

            # We do the s8 quantize in backend, the formula is qx = x / sc + 0
            qX_gpu = torch.quantize_per_tensor(
                X_gpu, X_scale, X_zero_point, dtype_inputs
            )
            print("fake qX_gpu:", X_gpu / X_scale)
            qW_gpu = torch.quantize_per_tensor(W_gpu, W_scale, 0, dtype_filters)
            print("qW_gpu:", qW_gpu)
            print("qW_gpu int_repr():", qW_gpu.int_repr())
            # prepack(weight, bias, stride, padding, output_padding, dilation, groups)
            packed_params_gpu = torch.ops.quantized.conv_transpose3d_prepack(
                qW_gpu, bias_gpu, (1, 1, 1), (0, 0, 0), (0, 0, 0), (1, 1, 1), 1
            )
            output_gpu_int8 = torch.ops.quantized.conv_transpose3d(
                qX_gpu, packed_params_gpu, Y_scale, Y_zero_point
            )
            print("xpu output result:", output_gpu_int8)
            print("xpu output result int_repr():", output_gpu_int8.int_repr())

            cpu_result = torch.dequantize(output_int8)
            gpu_result = torch.dequantize(output_gpu_int8)

            self.assertEqual(torch.dequantize(q_inputs), torch.dequantize(qX_gpu))
            self.assertEqual(torch.dequantize(q_filters), torch.dequantize(qW_gpu))
            self.assertEqual(cpu_result, gpu_result)
