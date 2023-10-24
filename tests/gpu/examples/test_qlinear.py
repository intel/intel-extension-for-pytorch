import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

import pytest
import platform


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        "fbgemm" not in torch.backends.quantized.supported_engines,
        reason="No qengine found. USE_FBGEMM=1 is needed for building pytorch",
    )
    def test_qlinear(self, dtype=torch.float):
        print(
            "Please open FBGEMM (PyTorch CPU INT8 default engine ) when build PyTorch, "
            "which will be referenced by GPU. Or you may meet runtime error like "
            "'Didn't find engine for operation quantized::linear_prepack'."
        )

        zero_point = 0

        dtype_inputs = torch.quint8
        dtype_filters = torch.qint8
        zp_filter = 0
        input_shape = [[5, 5], [3, 5, 5], [2, 3, 5, 5]]

        zp_vec = [0] if platform.system() == "Windows" else [0, 2]
        for shape in input_shape:
            for zp_in in zp_vec:
                inputs = torch.randn(shape)
                filters = torch.randn(5, 5)
                # bias = torch.randn(5)
                bias = None
                output_sc = 0.5
                output_zp = 0 if platform.system() == "Windows" else 2
                q_inputs = torch.quantize_per_tensor(inputs, 0.4, zp_in, dtype_inputs)
                q_filters = torch.quantize_per_tensor(
                    filters, 0.5, zp_filter, dtype_filters
                )

                packed_params = torch.ops.quantized.linear_prepack(q_filters, bias)
                output_int8 = torch.ops.quantized.linear(
                    q_inputs, packed_params, output_sc, output_zp
                )

                inputs_gpu = inputs.to("xpu")
                filters_gpu = filters.to("xpu")
                # bias_gpu = bias.to("xpu")
                bias_gpu = None

                q_inputs_gpu = torch.quantize_per_tensor(
                    inputs_gpu, 0.4, zp_in, dtype_inputs
                )
                q_filters_gpu = torch.quantize_per_tensor(
                    filters_gpu, 0.5, zp_filter, dtype_filters
                )

                packed_params_gpu = torch.ops.quantized.linear_prepack(
                    q_filters_gpu, bias_gpu
                )
                output_gpu_int8 = torch.ops.quantized.linear(
                    q_inputs_gpu, packed_params_gpu, output_sc, output_zp
                )
                self.assertEqual(
                    torch.dequantize(output_gpu_int8), torch.dequantize(output_int8)
                )
