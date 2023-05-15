import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

import pytest


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

        inputs = torch.randn(5, 5)
        filters = torch.randn(5, 5)
        bias = torch.randn(5)
        output_sc = 4.0

        q_inputs = torch.quantize_per_tensor(inputs, 0.4, zero_point, dtype_inputs)
        q_filters = torch.quantize_per_tensor(filters, 0.5, zero_point, dtype_filters)

        packed_params = torch.ops.quantized.linear_prepack(q_filters, bias)
        output_int8 = torch.ops.quantized.linear(q_inputs, packed_params, output_sc, 0)
        print(output_int8.int_repr())
        print(output_int8)

        inputs_gpu = inputs.to("xpu")
        filters_gpu = filters.to("xpu")
        bias_gpu = bias.to("xpu")

        q_inputs_gpu = torch.quantize_per_tensor(
            inputs_gpu, 0.4, zero_point, dtype_inputs
        )
        q_filters_gpu = torch.quantize_per_tensor(
            filters_gpu, 0.5, zero_point, dtype_filters
        )

        packed_params_gpu = torch.ops.quantized.linear_prepack(q_filters_gpu, bias_gpu)
        output_gpu_int8 = torch.ops.quantized.linear(
            q_inputs_gpu, packed_params_gpu, output_sc, 0
        )
        print(output_gpu_int8)
        # print(torch.round(output_gpu_int8 / output_sc))

        #  Align with FBGEMM, which output with UInt8.
        output_gpu_int8 = torch.quantize_per_tensor(
            output_gpu_int8, output_sc, zero_point, torch.quint8
        )
        print(output_gpu_int8.int_repr())

        self.assertEqual(
            output_int8.int_repr(), output_gpu_int8.int_repr(), rtol=2, atol=2
        )
