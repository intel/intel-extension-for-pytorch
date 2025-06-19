import itertools
import tempfile
import torch
import torch.nn as nn
from torch.testing import FileCheck
import copy
import unittest
from common_utils import TestCase

import intel_extension_for_pytorch as ipex
from test_ao_jit_llga_utils import JitLlgaTestCase, LLGA_FUSION_GROUP
from torch.testing._internal.common_utils import run_tests
from intel_extension_for_pytorch.quantization import (
    prepare,
    convert,
    dequantize_per_channel,
    dequantize_per_block,
    quantize_per_channel,
    quantize_per_block,
    WoqWeightDtype,
    WoqLowpMode,
    WoqWeightQScheme,
    WoqActQuantMode,
)
from intel_extension_for_pytorch.nn.modules.weight_only_quantization import (
    WeightOnlyQuantizedLinear,
    WoqWeightFormat,
)
import os

curpath = os.path.abspath(os.path.dirname(__file__))


class TestDefaultRecipe(JitLlgaTestCase):
    def test_quantized_op_int8_int8(self):
        # Test one op which only support INT8+INT8, if its
        # post op is not a quantifiable op, we need to make sure
        # it can also call in INT8 kernel by inset fake quant after it's output.
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = nn.Conv2d(2, 2, 1)
                self.pool = nn.MaxPool2d(1, 1)

            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                return x

        m = M()
        x = torch.rand(1, 2, 14, 14)

        graph = self.checkQuantizeTrace(m, [x], atol=2e-1)
        patterns = [
            [
                "aten::dequantize",
                "aten::dequantize",
                "aten::_convolution",
                "aten::quantize_per_tensor",
            ],
            ["aten::dequantize", "aten::max_pool2d", "aten::quantize_per_tensor"],
        ]
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
        self.checkPatterns(graph, patterns)

    def test_none_gemm_op_has_quantized_op_before(self):
        # For none-gemm op, if it's pre op is quantifiable op, fake quant will be inserted.
        # Given the following example, the quantization flow will be like:
        # q->dq->quantized_module->q->dq->flatten->q->dq.
        class M(nn.Module):
            def __init__(self, quantized_module):
                super(M, self).__init__()
                self.quantized_module = quantized_module

            def forward(self, x):
                x = self.quantized_module(x)
                x = x.flatten(1)
                # add a quantifiable op after flatten
                x = x.flatten(1)
                return x

        class conv_swish(nn.Module):
            def __init__(
                self,
            ):
                super(conv_swish, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 1)

            def forward(self, x):
                x = self.conv(x)
                y = x.sigmoid()
                z = torch.mul(x, y)
                return z

        class conv_eltwise(nn.Module):
            def __init__(
                self,
            ):
                super(conv_eltwise, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 1)

            def forward(self, x):
                x = self.conv(x)
                x = x.relu_()
                return x

        # TODO: test more quantized modules(especially for fused module).
        quantized_modules = [conv_swish(), conv_eltwise()]
        patterns = [
            [
                [
                    "aten::dequantize",
                    "aten::dequantize",
                    "aten::_convolution",
                    "aten::sigmoid",
                    "aten::mul",
                    "aten::quantize_per_tensor",
                ]
            ],
            [
                [
                    "aten::dequantize",
                    "aten::dequantize",
                    "aten::_convolution",
                    "aten::relu",
                    "aten::quantize_per_tensor",
                ]
            ],
        ]
        for quantized_module, pattern in zip(quantized_modules, patterns):
            m = M(quantized_module).eval()

            x = torch.rand(1, 2, 14, 14)

            graph = self.checkQuantizeTrace(m, [x], atol=2e-1)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            self.checkPatterns(graph, pattern)
            FileCheck().check("aten::dequantize").run(graph)

    def test_qconfig_mapping_for_static_quantization(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = nn.Conv2d(2, 2, 1)
                self.pool = nn.MaxPool2d(1, 1)

            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                return x

        m = M()
        x = torch.rand(1, 2, 14, 14)

        qconfig_mapping = ipex.quantization.default_static_qconfig_mapping
        graph = self.checkQuantizeTrace(m, [x], atol=2e-1, qconfig=qconfig_mapping)
        patterns = [
            [
                "aten::dequantize",
                "aten::dequantize",
                "aten::_convolution",
                "aten::quantize_per_tensor",
            ],
            ["aten::dequantize", "aten::max_pool2d", "aten::quantize_per_tensor"],
        ]
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
        self.checkPatterns(graph, patterns)

    def test_qconfig_mapping_for_dynamic_quantization(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear = nn.Linear(2, 2)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.linear(x)
                x = self.relu(x)
                return x

        m = M()
        x = torch.rand(1, 2)

        qconfig_mapping = ipex.quantization.default_dynamic_qconfig_mapping
        prepared_model = ipex.quantization.prepare(m, qconfig_mapping, x)
        converted_model = ipex.quantization.convert(prepared_model)
        assert hasattr(converted_model, "linear")
        assert isinstance(converted_model.linear, nn.quantized.dynamic.Linear)

    def test_check_model_obsever_has_run(self):
        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linears = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])

            def forward(self, x):
                for _, l in enumerate(self.linears):
                    x = l(x)
                return x

        class Mod(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([Block() for _ in range(2)])

            def forward(self, x):
                for _, b in enumerate(self.blocks):
                    x = b(x)
                return x

        check_model_obsever_has_run = (
            ipex.quantization._utils.check_model_obsever_has_run
        )
        m = Mod().eval()
        x = torch.rand(4, 4)
        qconfig_mapping = ipex.quantization.default_static_qconfig_mapping
        prepared_model = ipex.quantization.prepare(m, qconfig_mapping, x)
        assert not check_model_obsever_has_run(prepared_model)
        for _ in range(5):
            prepared_model(torch.rand(4, 4))
        assert check_model_obsever_has_run(prepared_model)
        with tempfile.NamedTemporaryFile() as fp:
            qconf_filename = fp.name
            prepared_model.save_qconf_summary(qconf_filename)
            # Observers are removed after save_qconf_summary
            assert not check_model_obsever_has_run(prepared_model)
            prepared_model.load_qconf_summary(qconf_filename)
            # Observers are added but not run yet after load_qconf_summary
            assert not check_model_obsever_has_run(prepared_model)
            for _ in range(5):
                prepared_model(torch.rand(4, 4))
            assert check_model_obsever_has_run(prepared_model)

    def test_none_example_input_for_quantization(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear = nn.Linear(2, 2)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.linear(x)
                x = self.relu(x)
                return x

        m = M()

        # Dynamic quant
        qconfig_mapping = ipex.quantization.default_dynamic_qconfig_mapping
        prepared_model = ipex.quantization.prepare(m, qconfig_mapping)
        converted_model = ipex.quantization.convert(prepared_model)
        assert hasattr(converted_model, "linear")
        assert isinstance(converted_model.linear, nn.quantized.dynamic.Linear)

        # Static quant
        qconfig_mapping = ipex.quantization.default_static_qconfig_mapping
        with self.assertRaises(AssertionError):
            prepared_model = ipex.quantization.prepare(m, qconfig_mapping)


class WeightOnlyQuantizationTester(TestCase):
    def test_weight_only_quantization(self):
        class M(nn.Module):
            def __init__(self, input_channel, output_channel, has_bias):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(feature, has_bias):
            model = M(feature[1], feature[2], has_bias)
            m = model.eval()
            data = torch.rand(1, feature[0], feature[1])
            weight = model.linear.weight
            weight_int8, w_scales, w_zero_points = quantize_per_channel(
                weight, WoqWeightDtype.INT8
            )
            weight_fp32 = dequantize_per_channel(
                weight_int8,
                w_scales,
                w_zero_points.int(),
                WoqWeightDtype.INT8,
                weight.shape,
            )
            if has_bias:
                bias = model.linear.bias
                output1 = torch.matmul(data, weight_fp32.T) + bias
            else:
                output1 = torch.matmul(data, weight_fp32.T)

            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping()
            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)
            with torch.no_grad():
                woq_model = convert(prepared_model)
                assert isinstance(woq_model.linear, WeightOnlyQuantizedLinear)
                assert (
                    woq_model.linear.weight is not None
                    and woq_model.linear.weight.dtype == torch.int8
                )

                output2 = woq_model(data)
                torch.testing.assert_close(output1, output2)

        shape_list = [
            [3, 31, 31],
            [4, 4096, 4096],
            [9, 4095, 4095],
            [9, 4096, 4096],
            [196, 4095, 16383],
            [1024, 512, 512],
        ]
        use_bias_list = [True, False]
        cases = itertools.product(shape_list, use_bias_list)
        for shape, use_bias in cases:
            test(shape, use_bias)

    def test_weight_only_quantization_autocast(self):
        class M(nn.Module):
            def __init__(self, input_channel, output_channel, has_bias):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(feature, has_bias, w_dtype):
            model = M(feature[1], feature[2], has_bias)
            m = model.eval()
            data = torch.rand(feature[0], feature[1])

            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=w_dtype
            )
            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)

            with torch.no_grad():
                weight = m.linear.weight
                weight_int8, w_scales, w_zero_points = quantize_per_channel(
                    weight, w_dtype
                )
                data_bf16 = data.bfloat16()
                data_fp16 = data_bf16.half()
                bias_fp32 = m.linear.bias
                # if M >= 32, compute in bf16
                # if M < 32, compute in fp32 or fp16. Depends on fp16 support.
                if feature[0] >= 32:
                    weight_bf16 = dequantize_per_channel(
                        weight_int8,
                        w_scales.bfloat16(),
                        w_zero_points.bfloat16(),
                        w_dtype,
                        weight.shape,
                    ).bfloat16()
                    output1 = torch.matmul(
                        data_bf16.float(), weight_bf16.float().T
                    ).float()
                    if has_bias:
                        output1 = output1 + bias_fp32
                    output1 = output1.bfloat16()
                    # For reference kernel
                    weight_bf16_ref = dequantize_per_channel(
                        weight_int8,
                        w_scales.float(),
                        w_zero_points.float(),
                        w_dtype,
                        weight.shape,
                    ).bfloat16()
                    output1_ref = torch.matmul(data_bf16, weight_bf16_ref.T)
                    if has_bias:
                        output1_ref = output1_ref + bias_fp32.bfloat16()
                    output1_ref = output1_ref.bfloat16()
                else:
                    weight_fp16 = dequantize_per_channel(
                        weight_int8,
                        w_scales.half(),
                        w_zero_points.half(),
                        w_dtype,
                        weight.shape,
                    )
                    output1_fp16 = torch.matmul(
                        data_fp16.float(), weight_fp16.float().T
                    ).half()
                    if has_bias:
                        output1_fp16 = output1_fp16 + bias_fp32.half()
                    output1_fp16 = output1_fp16.bfloat16()
                with torch.autocast(
                    device_type="cpu", enabled=True, dtype=torch.bfloat16
                ):
                    woq_model = convert(prepared_model)
                    assert isinstance(woq_model.linear, WeightOnlyQuantizedLinear)

                    woq_model = torch.jit.trace(woq_model, data)
                    woq_model = torch.jit.freeze(woq_model)
                    output2 = woq_model(data)
                    output2 = output2.bfloat16()
                if feature[0] < 32:
                    torch.testing.assert_close(
                        output1_fp16, output2, atol=0.01, rtol=0.1
                    )
                else:
                    # Use try...except to handle numeric differences between optimized and ref kernels
                    try:
                        torch.testing.assert_close(output1, output2)
                    except Exception:
                        torch.testing.assert_close(output1_ref, output2)

        shape_list = [
            [3, 31, 31],
            [4, 64, 64],
            [9, 128, 128],
            [196, 63, 255],
            [1024, 512, 512],
        ]
        use_bias_list = [True, False]
        w_dtype_list = [WoqWeightDtype.INT8, WoqWeightDtype.INT4]
        cases = itertools.product(shape_list, use_bias_list, w_dtype_list)
        for shape, use_bias, w_dtype in cases:
            test(shape, use_bias, w_dtype)

    def test_weight_only_quantization_non_fp32_model(self):
        class M(nn.Module):
            def __init__(self, input_channel, output_channel, has_bias):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

            def forward(self, x):
                return self.linear(x)

        shape_list = [
            [2, 24, 24],
            [8, 64, 64],
            [1024, 512, 512],
        ]
        use_bias_list = [True, False]
        w_dtype_list = [WoqWeightDtype.INT8, WoqWeightDtype.INT4]
        model_dtype_list = [torch.bfloat16, torch.half]
        cases = itertools.product(
            shape_list, use_bias_list, w_dtype_list, model_dtype_list
        )
        for shape, use_bias, w_dtype, model_dtype in cases:
            m = M(shape[1], shape[2], use_bias).to(model_dtype).eval()
            data = torch.rand(shape[0], shape[1])
            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=w_dtype
            )
            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)
            with torch.no_grad():
                woq_model = convert(prepared_model)
                # The following should pass
                woq_model(data)

    def test_weight_only_quantization_jit_save_load(self):
        class M(nn.Module):
            def __init__(self, input_channel, output_channel, has_bias):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(feature, has_bias, w_dtype):
            model = M(feature[1], feature[2], has_bias)
            m = model.eval()
            example_inputs = torch.rand(feature[0], feature[1])

            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=w_dtype
            )
            prepared_model = prepare(
                m, qconfig, example_inputs=example_inputs, inplace=False
            )
            with torch.no_grad():
                converted_model = convert(prepared_model)

                with tempfile.NamedTemporaryFile() as fp:
                    # save
                    with torch.no_grad():
                        traced_model = torch.jit.trace(converted_model, example_inputs)
                        traced_model = torch.jit.freeze(traced_model)
                        traced_model.save(fp.name)

                    # load
                    loaded_model = torch.jit.load(fp.name)

                    # Compare results of original model and loaded model
                    output_ref = traced_model(example_inputs)
                    output = loaded_model(example_inputs)
                    torch.testing.assert_close(output_ref, output)

        shape_list = [
            [3, 31, 31],
            [4, 4096, 4096],
            [4, 4096, 4080],
            [196, 4095, 16383],
            [1024, 512, 512],
        ]
        use_bias_list = [True, False]
        w_dtype_list = [WoqWeightDtype.INT8, WoqWeightDtype.INT4]
        cases = itertools.product(shape_list, use_bias_list, w_dtype_list)
        for shape, use_bias, w_dtype in cases:
            test(shape, use_bias, w_dtype)

    def test_weight_only_quantization_int4_weight(self):
        class M(nn.Module):
            def __init__(self, input_channel, output_channel, has_bias):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(feature, has_bias):
            model = M(feature[1], feature[2], has_bias)
            m = model.eval()
            data = torch.rand(feature[0], feature[1])
            weight = model.linear.weight
            weight_int4, w_scales, w_zero_points = quantize_per_channel(
                weight, WoqWeightDtype.INT4
            )
            weight_fp32 = dequantize_per_channel(
                weight_int4, w_scales, w_zero_points, WoqWeightDtype.INT4, weight.shape
            )
            if has_bias:
                bias = model.linear.bias
                output1 = torch.matmul(data, weight_fp32.T) + bias
            else:
                output1 = torch.matmul(data, weight_fp32.T)

            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=WoqWeightDtype.INT4
            )
            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)
            with torch.no_grad():
                woq_model = convert(prepared_model)
                assert isinstance(woq_model.linear, WeightOnlyQuantizedLinear)
                assert (
                    woq_model.linear.weight is not None
                    and woq_model.linear.weight.dtype == torch.uint8
                )

                output2 = woq_model(data)
                torch.testing.assert_close(output1, output2)

        shape_list = [
            [3, 31, 31],
            [4, 4096, 4096],
            [4, 4096, 4095],
            [9, 4095, 4095],
            [196, 4095, 16383],
            [1024, 512, 512],
        ]
        use_bias_list = [True, False]
        cases = itertools.product(shape_list, use_bias_list)
        for shape, use_bias in cases:
            test(shape, use_bias)

    def test_weight_only_quantization_nf4_weight(self):
        class M(nn.Module):
            def __init__(self, input_channel, output_channel, has_bias):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(feature, has_bias):
            model = M(feature[1], feature[2], has_bias)
            m = model.eval()
            data = torch.rand(feature[0], feature[1])
            weight = model.linear.weight
            weight_int4, w_scales, w_zero_points = quantize_per_channel(
                weight, WoqWeightDtype.NF4, sym_quant=True
            )
            weight_fp32 = dequantize_per_channel(
                weight_int4, w_scales, w_zero_points, WoqWeightDtype.NF4, weight.shape
            )
            if has_bias:
                bias = model.linear.bias
                output1 = torch.matmul(data, weight_fp32.T) + bias
            else:
                output1 = torch.matmul(data, weight_fp32.T)

            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=WoqWeightDtype.NF4
            )
            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)
            with torch.no_grad():
                woq_model = convert(prepared_model)
                assert isinstance(woq_model.linear, WeightOnlyQuantizedLinear)
                assert (
                    woq_model.linear.weight is not None
                    and woq_model.linear.weight.dtype == torch.uint8
                )

                output2 = woq_model(data)
                torch.testing.assert_close(output1, output2)

        shape_list = [
            [3, 31, 31],
            [4, 4096, 4096],
            [4, 4096, 4095],
            [9, 4095, 4095],
            [196, 4095, 4095],
            [1024, 512, 512],
        ]
        use_bias_list = [True, False]
        cases = itertools.product(shape_list, use_bias_list)
        for shape, use_bias in cases:
            test(shape, use_bias)

    def _test_weight_only_quantization_unary_fused_op_helper(
        self,
        post_op_module,
        fused_op,
    ):
        class Mod(nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = nn.Linear(64, 64, bias=bias)
                self.post_op = post_op_module

            def forward(self, x):
                return self.post_op(self.linear(x))

        weight_dtype_list = [
            WoqWeightDtype.INT8,
            WoqWeightDtype.INT4,
            WoqWeightDtype.NF4,
        ]
        bias_list = [False, True]
        bf16_list = [False, True]
        batch_size_list = [4, 1024]
        cases = itertools.product(
            weight_dtype_list, bias_list, bf16_list, batch_size_list
        )
        for w_dtype, bias, bf16, bs in cases:
            with torch.cpu.amp.autocast(
                enabled=bf16, dtype=torch.bfloat16 if bf16 else None
            ):
                model = Mod(bias).eval()
                data = torch.rand(bs, 64)
                qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                    weight_dtype=w_dtype, lowp_mode=2
                )
                prepared_model = prepare(
                    model, qconfig, example_inputs=data, inplace=False
                )
                with torch.no_grad():
                    woq_model = convert(prepared_model)
                    output1 = woq_model(data)
                    output2 = fused_op(
                        data, woq_model.linear._op_context.get_data_handle()
                    )
                    torch.testing.assert_close(
                        output1, output2.to(output1.dtype), atol=1e-2, rtol=1e-4
                    )

    def test_weight_only_quantization_gelu_fused_op(self):
        self._test_weight_only_quantization_unary_fused_op_helper(
            nn.GELU(), torch.ops.torch_ipex.woq_linear_gelu
        )

    def test_weight_only_quantization_new_gelu_fused_op(self):
        self._test_weight_only_quantization_unary_fused_op_helper(
            nn.GELU(approximate="tanh"), torch.ops.torch_ipex.woq_linear_new_gelu
        )

    def test_weight_only_quantization_relu_fused_op(self):
        self._test_weight_only_quantization_unary_fused_op_helper(
            nn.ReLU(), torch.ops.torch_ipex.woq_linear_relu
        )

    def test_weight_only_quantization_silu_fused_op(self):
        self._test_weight_only_quantization_unary_fused_op_helper(
            nn.SiLU(), torch.ops.torch_ipex.woq_linear_silu
        )

    def _test_weight_only_quantization_binary_fused_op_helper(
        self,
        num_extra_inputs,
        post_op,
        fused_op,
    ):
        class Mod(nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = nn.Linear(64, 64, bias=bias)

            def forward(self, x, others):
                y = self.linear(x)
                for o in others:
                    y = post_op(y, o)
                return y

        weight_dtype_list = [
            WoqWeightDtype.INT8,
            WoqWeightDtype.INT4,
            WoqWeightDtype.NF4,
        ]
        bias_list = [False, True]
        bf16_list = [False, True]
        batch_size_list = [4, 1024, 63]
        cases = itertools.product(
            weight_dtype_list, bias_list, bf16_list, batch_size_list
        )
        for w_dtype, bias, bf16, bs in cases:
            with torch.cpu.amp.autocast(
                enabled=bf16, dtype=torch.bfloat16 if bf16 else None
            ):
                model = Mod(bias).eval()
                data = torch.rand(bs, 64)
                extra_inputs = [torch.rand(bs, 64) for _ in range(num_extra_inputs)]
                qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                    weight_dtype=w_dtype, lowp_mode=2
                )
                prepared_model = prepare(
                    model, qconfig, example_inputs=data, inplace=False
                )
                with torch.no_grad():
                    woq_model = convert(prepared_model)
                    output1 = woq_model(data, extra_inputs)
                    output2 = fused_op(
                        data,
                        woq_model.linear._op_context.get_data_handle(),
                        extra_inputs,
                    )
                    torch.testing.assert_close(
                        output1, output2.to(output1.dtype), atol=1.5e-2, rtol=1e-3
                    )

    def test_weight_only_quantization_add_fused_op(self):
        # linear - add
        num_extra_inputs = 1
        self._test_weight_only_quantization_binary_fused_op_helper(
            num_extra_inputs,
            torch.add,
            torch.ops.torch_ipex.woq_linear_add,
        )
        # linear - add - add
        num_extra_inputs = 2
        self._test_weight_only_quantization_binary_fused_op_helper(
            num_extra_inputs,
            torch.add,
            torch.ops.torch_ipex.woq_linear_add_add,
        )

    def test_weight_only_quantization_mul_fused_op(self):
        num_extra_inputs = 1
        self._test_weight_only_quantization_binary_fused_op_helper(
            num_extra_inputs,
            torch.mul,
            torch.ops.torch_ipex.woq_linear_mul,
        )

    def test_weight_only_quantization_lowp_mode_functionality(self):
        from intel_extension_for_pytorch.quantization import WoqLowpMode

        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, x):
                return self.linear(x)

        data = torch.rand(4, 64)
        m = M()
        for mode in [
            WoqLowpMode.NONE,
            WoqLowpMode.FP16,
            WoqLowpMode.BF16,
            WoqLowpMode.INT8,
        ]:
            kwargs = {"lowp_mode": mode}
            if mode == WoqLowpMode.INT8:
                kwargs["weight_dtype"] = WoqWeightDtype.INT4
            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(**kwargs)
            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)
            with torch.no_grad():
                woq_model = convert(prepared_model)
                woq_model(data)
                assert (
                    hasattr(woq_model.linear, "_lowp_mode")
                    and woq_model.linear._lowp_mode == mode
                ), "Weight-only quantization: low precision gemm flag is not correctly set"

    def test_weight_only_quantization_int8_lowp_mode_correctness(self):
        from intel_extension_for_pytorch.quantization import WoqLowpMode

        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(64, 128)

            def forward(self, x):
                return self.linear(x)

        m = M()

        lowp_mode_list = [WoqLowpMode.NONE, WoqLowpMode.FP16, WoqLowpMode.BF16]
        act_dtype_list = [torch.bfloat16, torch.half]
        compute_dtype_list = [None, torch.half, torch.bfloat16]
        batch_size_list = [4, 1024]
        cases = itertools.product(lowp_mode_list, act_dtype_list, batch_size_list)
        # lowp_mode does not affect weight observer for int8
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping()
        weight = copy.deepcopy(m.linear.weight)
        w_dtype = qconfig.global_qconfig.weight_dtype
        weight_int8, w_scales, w_zps = quantize_per_channel(weight, w_dtype)
        weight_fp32 = dequantize_per_channel(weight_int8, w_scales, w_zps, w_dtype)
        bias_fp32 = copy.deepcopy(m.linear.bias)
        for lowp_mode, act_dtype, bs in cases:
            # When lowp_mode=BF16, only case of batch size >= 32 uses BF16.
            if lowp_mode == WoqLowpMode.BF16 and bs < 32:
                continue
            data = torch.rand(bs, 64)
            if lowp_mode == WoqLowpMode.NONE:
                compute_dtype_list[0] = act_dtype
            compute_dtype = compute_dtype_list[int(lowp_mode)]
            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                lowp_mode=lowp_mode,
                weight_dtype=WoqWeightDtype.INT8,
            )
            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)
            with torch.no_grad():
                woq_model = convert(prepared_model)
                y = woq_model(data.to(act_dtype))
                weight_for_compute = weight_fp32.to(compute_dtype).float()
                act_for_compute = data.to(act_dtype).to(compute_dtype).float()
                bias_for_compute = bias_fp32.to(compute_dtype).float()
                y_ref = act_for_compute @ weight_for_compute.T + bias_for_compute
                y_ref = y_ref.to(act_dtype)
                torch.testing.assert_close(y, y_ref, atol=0.005, rtol=0.01)

    def _fakequant_by_group(self, t, quant_a_mode, groupsize):
        assert quant_a_mode >= 0 and quant_a_mode <= 3
        if quant_a_mode == 0:
            obs = torch.ao.quantization.MinMaxObserver(torch.quint8)
            obs(t)
            scale, zero_point = obs.calculate_qparams()
            return (
                torch.quantize_per_tensor(
                    t.to(torch.float), scale, zero_point, torch.quint8
                )
                .dequantize()
                .to(t.dtype)
            )
        orig_shape = t.shape
        if t.shape[-1] % groupsize:
            pad_len = t.shape[-1] // groupsize * groupsize + groupsize - t.shape[-1]
            t = torch.nn.functional.pad(t, (0, pad_len), value=0)
        grouped = t.view(-1, t.shape[-1] // groupsize, groupsize)
        if quant_a_mode == 1:
            grouped_min = grouped.min(dim=-1)[0].min(dim=0)[0]
            grouped_max = grouped.max(dim=-1)[0].max(dim=0)[0]
        elif quant_a_mode == 2:
            grouped_min = grouped.min(dim=-1)[0].min(dim=1)[0]
            grouped_max = grouped.max(dim=-1)[0].max(dim=1)[0]
        else:
            grouped_min = grouped.min(dim=-1)[0]
            grouped_max = grouped.max(dim=-1)[0]
        min = grouped_min
        max = grouped_max
        eps = torch.tensor([torch.finfo(torch.float32).eps])
        scales = (max - min) / 255
        scales = torch.max(scales, eps)
        zps = -torch.round(min / scales)
        if quant_a_mode == 1:
            qt = torch.clamp(
                torch.round(grouped / scales.unsqueeze(1)) + zps.unsqueeze(1),
                min=0,
                max=255,
            )
            out = (
                ((qt - zps.unsqueeze(1)) * scales.unsqueeze(1))
                .to(t.dtype)
                .view(t.shape)
            )
            if orig_shape != out.shape:
                out = out[: orig_shape[0], : orig_shape[1]].contiguous()
            return out
        elif quant_a_mode == 2:
            qt = torch.clamp(
                torch.round(grouped / scales.unsqueeze(1).unsqueeze(2))
                + zps.unsqueeze(1).unsqueeze(2),
                min=0,
                max=255,
            )
            out = (
                (
                    (qt - zps.unsqueeze(1).unsqueeze(2))
                    * scales.unsqueeze(1).unsqueeze(2)
                )
                .to(t.dtype)
                .view(t.shape)
            )
            if orig_shape != out.shape:
                out = out[: orig_shape[0], : orig_shape[1]].contiguous()
            return out
        else:
            qt = torch.clamp(
                torch.round(grouped / scales.unsqueeze(-1)) + zps.unsqueeze(-1),
                min=0,
                max=255,
            )
            out = (
                ((qt - zps.unsqueeze(-1)) * scales.unsqueeze(-1))
                .to(t.dtype)
                .view(t.shape)
            )
            if orig_shape != out.shape:
                out = out[: orig_shape[0], : orig_shape[1]].contiguous()
            return out

    def _fakequant_by_group_sym(self, t, quant_a_mode, groupsize):
        assert quant_a_mode >= 4 and quant_a_mode <= 7
        if quant_a_mode == 4:
            obs = torch.ao.quantization.MinMaxObserver(
                torch.qint8, qscheme=torch.per_tensor_symmetric
            )
            obs(t)
            scale, zero_point = obs.calculate_qparams()
            return (
                torch.quantize_per_tensor(
                    t.to(torch.float), scale, zero_point, torch.qint8
                )
                .dequantize()
                .to(t.dtype)
            )
        orig_shape = t.shape
        if t.shape[-1] % groupsize:
            pad_len = t.shape[-1] // groupsize * groupsize + groupsize - t.shape[-1]
            t = torch.nn.functional.pad(t, (0, pad_len), value=0)
        grouped = t.view(-1, t.shape[-1] // groupsize, groupsize)
        if quant_a_mode == 5:
            grouped_min = grouped.min(dim=-1)[0].min(dim=0)[0]
            grouped_max = grouped.max(dim=-1)[0].max(dim=0)[0]
        elif quant_a_mode == 6:
            grouped_min = grouped.min(dim=-1)[0].min(dim=1)[0]
            grouped_max = grouped.max(dim=-1)[0].max(dim=1)[0]
        else:
            grouped_min = grouped.min(dim=-1)[0]
            grouped_max = grouped.max(dim=-1)[0]
        min = grouped_min
        max = grouped_max
        eps = torch.tensor([torch.finfo(torch.float32).eps])
        scales = torch.max(torch.abs(max), torch.abs(min)) / 127
        scales = torch.max(scales, eps)
        if quant_a_mode == 5:
            qt = torch.clamp(
                torch.round(grouped / scales.unsqueeze(1)),
                min=-128,
                max=127,
            )
            out = ((qt) * scales.unsqueeze(1)).to(t.dtype).view(t.shape)
            if orig_shape != out.shape:
                out = out[: orig_shape[0], : orig_shape[1]].contiguous()
            return out
        elif quant_a_mode == 6:
            qt = torch.clamp(
                torch.round(grouped / scales.unsqueeze(1).unsqueeze(2)),
                min=-128,
                max=127,
            )
            out = ((qt) * scales.unsqueeze(1).unsqueeze(2)).to(t.dtype).view(t.shape)
            if orig_shape != out.shape:
                out = out[: orig_shape[0], : orig_shape[1]].contiguous()
            return out
        else:
            qt = torch.clamp(
                torch.round(grouped / scales.unsqueeze(-1)),
                min=-128,
                max=127,
            )
            out = ((qt) * scales.unsqueeze(-1)).to(t.dtype).view(t.shape)
            if orig_shape != out.shape:
                out = out[: orig_shape[0], : orig_shape[1]].contiguous()
            return out

    def test_weight_only_quantization_act_quant_mode(self):

        class Mod(nn.Module):
            def __init__(self, has_bias):
                super(Mod, self).__init__()
                self.linear = torch.nn.Linear(K, N, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(has_bias, act_quant_mode, M):
            dtype = torch.bfloat16
            model = Mod(has_bias)
            m = model.eval()
            m2 = copy.deepcopy(m)
            data = torch.rand(M, K) * 0.5
            qconfig_mapping = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=WoqWeightDtype.INT4,
                lowp_mode=WoqLowpMode.INT8,
                act_quant_mode=act_quant_mode,
            )
            fake_quant_x = self._fakequant_by_group(data, act_quant_mode, groupsize)
            prepared_model = prepare(m2, qconfig_mapping, inplace=True)
            with torch.no_grad(), torch.autocast(
                device_type="cpu", enabled=True, dtype=dtype
            ):
                woq_model = convert(prepared_model)
                # Behavior of WOQ Linear to simulate:
                # Quantize weight to int4 by float qparams at quantization time
                # Quantize activation to int8 at runtime
                # Convert weight and its zero points to INT8 for computation
                qw = woq_model.linear._op_context.to_public(
                    woq_model.linear._op_context.get_weight()
                )
                w_scales = woq_model.linear._op_context.get_scales()
                w_zero_points = woq_model.linear._op_context.get_zero_points()
                w = copy.deepcopy(m.linear.weight.data)
                qw, _, _ = quantize_per_channel(
                    w, WoqWeightDtype.INT4, w_scales, w_zero_points
                )
                fake_quant_w = dequantize_per_channel(
                    qw, w_scales, w_zero_points.int(), WoqWeightDtype.INT4, w.shape
                )
                m.linear.weight.data = fake_quant_w
                y_ref = m(fake_quant_x).to(dtype)
                y = woq_model(data)
                try:
                    torch.testing.assert_close(y, y_ref, atol=1e-2 * 5, rtol=1e-1 * 2)
                except Exception:
                    # The fallback kernel does not support act quant mode
                    # It computes in fp32 by dequantizing weight.
                    fake_quant_w = qw.dequantize()
                    y_ref = data @ fake_quant_w.T + (m.linear.bias if has_bias else 0)
                    y_ref = y_ref.to(dtype)
                    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-1)

        N, K = 64, 512
        groupsize = 64
        has_bias_list = [False, True]
        quant_mode_list = [0, 1, 2, 3]
        batch_size_list = [4, 1024]
        cases = itertools.product(has_bias_list, quant_mode_list, batch_size_list)
        for has_bias, quant_mode, M in cases:
            test(has_bias, quant_mode, M)

    def test_weight_only_quantization_act_quant_sym_mode(self):

        class Mod(nn.Module):
            def __init__(self, has_bias, K, N):
                super(Mod, self).__init__()
                self.linear = torch.nn.Linear(K, N, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test_sym(has_bias, act_quant_mode, shape):
            dtype = torch.bfloat16
            model = Mod(has_bias, shape[1], shape[2])
            m = model.eval()
            m2 = copy.deepcopy(m)
            data = torch.randn(shape[0], shape[1]) * 0.5
            qconfig_mapping = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=WoqWeightDtype.INT4,
                lowp_mode=WoqLowpMode.INT8,
                act_quant_mode=act_quant_mode,
            )
            fake_quant_x_sym = self._fakequant_by_group_sym(
                data, act_quant_mode, groupsize
            )
            prepared_model = prepare(m2, qconfig_mapping, inplace=True)
            with torch.no_grad(), torch.autocast(
                device_type="cpu", enabled=True, dtype=dtype
            ):
                woq_model = convert(prepared_model)
                # Behavior of WOQ Linear to simulate:
                # Quantize weight to int4 by float qparams at quantization time
                # Quantize activation to int8 at runtime
                # Convert weight and its zero points to INT8 for computation
                qw = woq_model.linear._op_context.to_public(
                    woq_model.linear._op_context.get_weight()
                )
                w_scales = woq_model.linear._op_context.get_scales()
                w_zero_points = woq_model.linear._op_context.get_zero_points()
                w = copy.deepcopy(m.linear.weight.data)

                qw, _, _ = quantize_per_channel(
                    w, WoqWeightDtype.INT4, w_scales, w_zero_points
                )
                fake_quant_w = dequantize_per_channel(
                    qw, w_scales, w_zero_points.int(), WoqWeightDtype.INT4, w.shape
                )
                m.linear.weight.data = fake_quant_w
                y_ref = m(fake_quant_x_sym).to(dtype)
                y = woq_model(data)
                try:
                    torch.testing.assert_close(y, y_ref, atol=1e-2 * 5, rtol=1e-1 * 2)
                except Exception:
                    # The fallback kernel does not support act quant mode
                    # It computes in fp32 by dequantizing weight.
                    fake_quant_w = qw.dequantize()
                    y_ref = data @ fake_quant_w.T + (m.linear.bias if has_bias else 0)
                    y_ref = y_ref.to(dtype)
                    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-1)

        groupsize = 64
        shape_list = [
            [3, 31, 31],
            [4, 4096, 4096],
            [4, 4096, 4095],
            [9, 4095, 4095],
            [196, 4095, 4095],
            [1024, 512, 512],
        ]
        has_bias_list = [False, True]
        quant_mode_sym_list = [4, 5, 6, 7]
        cases_sym = itertools.product(has_bias_list, quant_mode_sym_list, shape_list)
        for has_bias, quant_mode, shape in cases_sym:
            test_sym(has_bias, quant_mode, shape)

    def test_weight_only_quantization_group_size(self):
        class Mod(nn.Module):
            def __init__(self, ic, oc, has_bias):
                super(Mod, self).__init__()
                self.linear = torch.nn.Linear(ic, oc, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(shape, has_bias, act_quant_mode, group_size, w_dtype):
            dtype = torch.bfloat16
            model = Mod(shape[1], shape[2], has_bias)
            m = model.eval()
            m2 = copy.deepcopy(m)
            data = torch.rand(shape[0], shape[1])
            if w_dtype == WoqWeightDtype.INT4:
                lowp_mode = WoqLowpMode.INT8
            else:
                lowp_mode = WoqLowpMode.BF16
            if group_size == -1 and act_quant_mode != 0:
                # these cases are covered by another test case for act_quant_mode
                return
            qconfig_mapping = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=w_dtype,
                lowp_mode=lowp_mode,
                act_quant_mode=act_quant_mode,
                group_size=group_size,
            )
            fake_quant_x = self._fakequant_by_group(data, act_quant_mode, group_size)
            prepared_model = prepare(m2, qconfig_mapping, inplace=True)
            with torch.no_grad(), torch.autocast(
                device_type="cpu", enabled=True, dtype=dtype
            ):
                woq_model = convert(prepared_model)
                # Behavior of WOQ Linear to simulate:
                # Quantize weight to int4 by float qparams at quantization time
                # Quantize activation to int8 at runtime
                # Convert weight and its zero points to INT8 for computation
                w = copy.deepcopy(m.linear.weight.data)
                sym_quant = w_dtype == WoqWeightDtype.NF4
                if group_size == -1:
                    qw, w_scales, w_zero_points = quantize_per_channel(
                        w, w_dtype, None, None, sym_quant
                    )
                    fake_quant_w = dequantize_per_channel(
                        qw, w_scales, w_zero_points, w_dtype, w.shape
                    )
                else:
                    qw, w_scales, w_zero_points = quantize_per_block(
                        w, w_dtype, group_size, None, None, sym_quant
                    )
                    fake_quant_w = dequantize_per_block(
                        qw,
                        w_scales,
                        w_zero_points,
                        w_dtype,
                        group_size,
                        weight_shape=w.shape,
                    )
                m.linear.weight.data = fake_quant_w
                y_ref = m(fake_quant_x).to(dtype)
                y = woq_model(data)
                try:
                    torch.testing.assert_close(y, y_ref, atol=1e-2 * 5, rtol=1e-1 * 2)
                except Exception:
                    # The fallback kernel does not support act quant mode
                    # It computes in fp32 by dequantizing weight.
                    # fake_quant_w = qw.dequantize()
                    y_ref = data @ fake_quant_w.T + (m.linear.bias if has_bias else 0)
                    y_ref = y_ref.to(dtype)
                    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-1)

        MKN_list = [
            (4, 64, 128),
            (4, 32, 127),
            (9, 31, 256),
            (4, 144, 64),
            (16, 256, 256),
        ]
        has_bias_list = [False, True]
        quant_mode_list = [0, 1, 2, 3]
        group_size_list = [-1, 32, 64, 128]
        weight_dtype = [WoqWeightDtype.INT8, WoqWeightDtype.INT4, WoqWeightDtype.NF4]
        cases = itertools.product(
            MKN_list, has_bias_list, quant_mode_list, group_size_list, weight_dtype
        )
        for shape, has_bias, act_quant_mode, group_size, w_dtype in cases:
            test(shape, has_bias, act_quant_mode, group_size, w_dtype)

    def test_compute_with_g_idx(self):
        class Mod(nn.Module):
            def __init__(self, ic, oc, has_bias):
                super(Mod, self).__init__()
                self.linear = torch.nn.Linear(ic, oc, has_bias)

            def forward(self, x):
                return self.linear(x)

        shape_list = [[1, 32, 32], [16, 64, 64], [32, 128, 128]]
        group_size_list = [4, 16]
        lowp_mode_list = [
            WoqLowpMode.NONE,
            WoqLowpMode.FP16,
            WoqLowpMode.BF16,
            WoqLowpMode.INT8,
        ]
        compute_dtype_list = [torch.float, torch.half, torch.bfloat16, torch.float]
        cases = itertools.product(shape_list, group_size_list, lowp_mode_list)
        for shape, group_size, lowp_mode in cases:
            bs, ic, oc = shape
            n_groups = ic // group_size
            int4_weight = torch.randint(0, 15, (oc, ic), dtype=torch.uint8)
            packed_weight = (
                int4_weight[:, 1::2]
                .bitwise_left_shift(4)
                .bitwise_or_(int4_weight[:, ::2])
            )
            scales = torch.rand((oc, n_groups), dtype=torch.half) * 0.1
            zeros = torch.randint(6, 9, (oc, n_groups), dtype=torch.uint8)
            packed_zeros = torch.zeros(
                (oc, (n_groups * 4 + 32 - 1) // 32), dtype=torch.int32
            )
            for i in range(n_groups):
                packed_zeros[:, i // 8] = packed_zeros[:, i // 8].bitwise_or_(
                    zeros[:, i].int().bitwise_left_shift(4 * (i % 8))
                )
            g_idx = torch.arange(0, n_groups).to(torch.int64).repeat(group_size)
            x = torch.randn((bs, ic), dtype=torch.float)
            compute_dtype = compute_dtype_list[int(lowp_mode)]
            for has_bias in [True, False]:
                # woq path
                m = Mod(ic=ic, oc=oc, has_bias=has_bias)
                b = m.linear.bias.detach() if has_bias else None
                b = b.to(compute_dtype).float() if has_bias else None
                qconfig_mapping = (
                    ipex.quantization.get_weight_only_quant_qconfig_mapping(
                        weight_dtype=WoqWeightDtype.INT4,
                        lowp_mode=lowp_mode,
                        act_quant_mode=ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
                        group_size=group_size,
                    )
                )
                woq_m = copy.deepcopy(m)
                woq_m.linear.qconfig = qconfig_mapping.global_qconfig
                woq_m.linear = WeightOnlyQuantizedLinear.from_float_and_qweight(
                    woq_m.linear,
                    packed_weight,
                    WoqWeightDtype.INT4,
                    scales,
                    packed_zeros,
                    b,
                    group_size=group_size,
                    g_idx=g_idx,
                )
                y = woq_m(x)

                # ref path
                if lowp_mode == WoqLowpMode.INT8:
                    # shuffle x so that each group is contiguous for fake quantization
                    # then shuffle back after fake quantization
                    x_shuffled = torch.empty_like(x)
                    for g in range(n_groups):
                        indices = (g_idx == g).nonzero().flatten()
                        for i in range(indices.numel()):
                            x_shuffled[:, g * group_size + i] = x[:, indices[i]]
                    fqx_shuffled = self._fakequant_by_group(
                        x_shuffled, 1, group_size
                    ).float()
                    fqx = torch.empty_like(fqx_shuffled)
                    for g in range(n_groups):
                        indices = (g_idx == g).nonzero().flatten()
                        for i in range(indices.numel()):
                            fqx[:, indices[i]] = fqx_shuffled[:, g * group_size + i]
                else:
                    fqx = x
                fqx = fqx.to(compute_dtype).float()
                scales_expanded = scales.repeat(1, group_size).to(compute_dtype).float()
                zeros_expanded = zeros.repeat(1, group_size).to(compute_dtype).float()
                dqw = (int4_weight.to(torch.float) - zeros_expanded) * scales_expanded
                y_ref = torch.nn.functional.linear(fqx, dqw, bias=b)
                y_ref_2 = torch.nn.functional.linear(x, dqw, bias=b)

                # check results
                atol = 1e-4
                rtol = 1e-5
                if lowp_mode == WoqLowpMode.FP16:
                    atol = 5e-2
                    rtol = 1e-3
                elif lowp_mode == WoqLowpMode.BF16:
                    atol = 1e-1
                    rtol = 1e-3
                try:
                    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)
                except Exception:
                    # In IPEX CI, UT will run with different ISA
                    # This check is for the ref kernel, where x is not quantized
                    torch.testing.assert_close(y, y_ref_2, atol=atol, rtol=rtol)

    def test_unpack_with_g_idx(self):
        class Mod(nn.Module):
            def __init__(self, ic, oc, has_bias):
                super(Mod, self).__init__()
                self.linear = torch.nn.Linear(ic, oc, has_bias)

            def forward(self, x):
                return self.linear(x)

        shape_list = [[64, 64], [256, 256]]
        group_size_list = [4, 16]
        lowp_mode_list = [WoqLowpMode.BF16, WoqLowpMode.INT8]
        cases = itertools.product(shape_list, group_size_list, lowp_mode_list)
        for shape, group_size, lowp_mode in cases:
            ic, oc = shape
            n_groups = ic // group_size
            int4_weight = torch.randint(0, 15, (oc, ic), dtype=torch.uint8)
            packed_weight = (
                int4_weight[:, 1::2]
                .bitwise_left_shift(4)
                .bitwise_or_(int4_weight[:, ::2])
            )
            scales = torch.randn((oc, n_groups), dtype=torch.half)
            zeros = torch.randint(6, 9, (oc, n_groups), dtype=torch.uint8)
            packed_zeros = torch.zeros(
                (oc, (n_groups * 4 + 32 - 1) // 32), dtype=torch.int32
            )
            for i in range(n_groups):
                packed_zeros[:, i // 8] = packed_zeros[:, i // 8].bitwise_or_(
                    zeros[:, i].int().bitwise_left_shift(4 * (i % 8))
                )
            g_idx = torch.arange(0, n_groups).to(torch.int64).repeat(group_size)
            for has_bias in [True, False]:
                m = Mod(ic=ic, oc=oc, has_bias=has_bias)
                b = m.linear.bias.detach() if has_bias else None
                qconfig_mapping = (
                    ipex.quantization.get_weight_only_quant_qconfig_mapping(
                        weight_dtype=WoqWeightDtype.INT4,
                        lowp_mode=lowp_mode,
                        act_quant_mode=ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
                        group_size=group_size,
                    )
                )
                scales_expanded = scales.repeat(1, group_size)
                zeros_expanded = zeros.repeat(1, group_size)
                # path with g_idx
                woq_m = copy.deepcopy(m)
                woq_m.linear.qconfig = qconfig_mapping.global_qconfig
                woq_m.linear = WeightOnlyQuantizedLinear.from_float_and_qweight(
                    woq_m.linear,
                    packed_weight,
                    WoqWeightDtype.INT4,
                    scales,
                    packed_zeros,
                    b,
                    group_size=group_size,
                    g_idx=g_idx,
                )
                qw = woq_m.linear._op_context.to_public(
                    woq_m.linear._op_context.get_weight()
                )
                qw_uint8 = torch.empty(qw.size(0), qw.size(1) * 2, dtype=qw.dtype)
                qw_uint8[:, ::2] = qw.bitwise_and(0xF)
                qw_uint8[:, 1::2] = qw.bitwise_right_shift(4)
                dqw = (qw_uint8.to(torch.float) - zeros_expanded) * scales_expanded
                # reference: without g_idx
                woq_m_2 = copy.deepcopy(m)
                woq_m_2.linear.qconfig = qconfig_mapping.global_qconfig
                woq_m_2.linear = WeightOnlyQuantizedLinear.from_float_and_qweight(
                    woq_m_2.linear,
                    packed_weight,
                    WoqWeightDtype.INT4,
                    scales,
                    packed_zeros,
                    b,
                    group_size=group_size,
                    g_idx=None,
                )
                qw_2 = woq_m_2.linear._op_context.to_public(
                    woq_m_2.linear._op_context.get_weight()
                )
                qw_uint8_2 = torch.empty(qw_2.size(0), qw.size(1) * 2, dtype=qw_2.dtype)
                qw_uint8_2[:, ::2] = qw_2.bitwise_and(0xF)
                qw_uint8_2[:, 1::2] = qw_2.bitwise_right_shift(4)
                dqw_2 = (qw_uint8_2.to(torch.float) - zeros_expanded) * scales_expanded
                # Dequantized weights should be close
                torch.testing.assert_close(dqw, dqw_2)

    def test_g_idx_tp(self):
        """
        Test g_idx with autoTP and weight is split by K. For lowp_mode != INT8.
        Since it's complicated to write a test case for deepspeed,
        here we do the sharding manually to simulate the behavior.
        """

        class Mod(nn.Module):
            def __init__(self, ic, oc):
                super(Mod, self).__init__()
                # set bias to false and we add it ourselves
                self.linear = torch.nn.Linear(ic, oc, bias=False)

            def forward(self, x):
                return self.linear(x)

        shape_list = [[1, 32, 32], [32, 128, 128], [128, 128, 128], [1024, 128, 128]]
        group_size_list = [4, 16]
        lowp_mode_list = [WoqLowpMode.NONE, WoqLowpMode.FP16, WoqLowpMode.BF16]
        has_bias_list = [False, True]
        cases = itertools.product(
            shape_list, group_size_list, lowp_mode_list, has_bias_list
        )
        for shape, group_size, lowp_mode, has_bias in cases:
            bs, ic, oc = shape
            n_groups = ic // group_size
            dtype = torch.float
            if lowp_mode == WoqLowpMode.BF16:
                dtype = torch.bfloat16
            elif lowp_mode == WoqLowpMode.FP16:
                dtype = torch.half
            int4_weight = torch.randint(0, 15, (oc, ic), dtype=torch.uint8)
            packed_weight = (
                int4_weight[:, 1::2]
                .bitwise_left_shift(4)
                .bitwise_or_(int4_weight[:, ::2])
            )
            scales = torch.rand((oc, n_groups), dtype=torch.half) * 0.1
            zeros = torch.randint(6, 9, (oc, n_groups), dtype=torch.uint8)
            packed_zeros = torch.zeros(
                (oc, (n_groups * 4 + 32 - 1) // 32), dtype=torch.int32
            )
            for i in range(n_groups):
                packed_zeros[:, i // 8] = packed_zeros[:, i // 8].bitwise_or_(
                    zeros[:, i].int().bitwise_left_shift(4 * (i % 8))
                )
            g_idx = torch.arange(0, n_groups).to(torch.int).repeat(group_size)
            x = torch.randn((bs, ic), dtype=torch.float)

            # woq path
            b = torch.randn(oc).to(dtype) if has_bias else None
            qconfig_mapping = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=WoqWeightDtype.INT4,
                lowp_mode=lowp_mode,
                group_size=group_size,
            )
            m = Mod(ic=ic // 2, oc=oc)
            woq_m_0 = copy.deepcopy(m)
            woq_m_0.linear.qconfig = qconfig_mapping.global_qconfig
            packed_weight_0 = packed_weight[
                :, : packed_weight.shape[1] // 2
            ].contiguous()
            g_idx_0 = g_idx[: g_idx.shape[0] // 2]
            woq_m_0.linear = (
                ipex.nn.modules.WeightOnlyQuantizedLinear.from_float_and_qweight(
                    woq_m_0.linear,
                    packed_weight_0,
                    WoqWeightDtype.INT4,
                    scales,
                    packed_zeros,
                    None,  # bias
                    group_size=group_size,
                    g_idx=g_idx_0,
                )
            )
            woq_m_1 = copy.deepcopy(m)
            woq_m_1.linear.qconfig = qconfig_mapping.global_qconfig
            packed_weight_1 = packed_weight[
                :, packed_weight.shape[1] // 2 :
            ].contiguous()
            g_idx_1 = g_idx[g_idx.shape[0] // 2 :]
            woq_m_1.linear = WeightOnlyQuantizedLinear.from_float_and_qweight(
                woq_m_1.linear,
                packed_weight_1,
                WoqWeightDtype.INT4,
                scales,
                packed_zeros,
                None,  # bias
                group_size=group_size,
                g_idx=g_idx_1,
            )
            x_0 = x[:, : x.shape[1] // 2].contiguous()
            y_0 = woq_m_0(x_0.to(dtype))
            x_1 = x[:, x.shape[1] // 2 :].contiguous()
            y_1 = woq_m_1(x_1.to(dtype))
            y = y_0 + y_1 + b if has_bias else y_0 + y_1

            # ref path
            scales_expanded = scales.repeat(1, group_size)
            zeros_expanded = zeros.repeat(1, group_size)
            dqw = (int4_weight.float() - zeros_expanded) * scales_expanded
            dqw = dqw.to(dtype)
            x = x.to(dtype)
            y_ref = torch.nn.functional.linear(x, dqw, bias=b).to(dtype)

            # check results
            atol = 1e-4
            rtol = 1e-5
            if lowp_mode == WoqLowpMode.BF16:
                atol = 1e-1
                rtol = 1e-3
            elif lowp_mode == WoqLowpMode.FP16:
                atol = 1.5e-2
                rtol = 1e-3
            torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)

    def test_weight_only_quantization_weight_for_first_token(self):
        class M(nn.Module):
            def __init__(self, input_channel, output_channel, has_bias):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(feature, has_bias, w_dtype, lowp_mode, group_size):
            if lowp_mode == WoqLowpMode.INT8 and w_dtype == WoqWeightDtype.NF4:
                # not supported yet
                return
            model = M(feature[1], feature[2], has_bias)
            m = model.to(torch.bfloat16).eval()
            data = torch.rand(feature[0], feature[1])

            qconfig_ref = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=w_dtype,
                lowp_mode=lowp_mode,
                group_size=group_size,
            )
            prepared_model_ref = prepare(
                m, qconfig_ref, example_inputs=data, inplace=False
            )
            from intel_extension_for_pytorch.utils.weight_only_quantization import (
                _woq_enable_weight_cache_for_large_batch,
            )

            qconfig = _woq_enable_weight_cache_for_large_batch(qconfig_ref)
            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)

            with torch.no_grad(), torch.autocast(
                device_type="cpu", enabled=True, dtype=torch.bfloat16
            ):
                woq_model_ref = convert(prepared_model_ref)
                woq_model_ref = torch.jit.trace(woq_model_ref, data)
                woq_model_ref = torch.jit.freeze(woq_model_ref)
                woq_model = convert(prepared_model)
                woq_model = torch.jit.trace(woq_model, data)
                woq_model = torch.jit.freeze(woq_model)
                out_ref = woq_model_ref(data).bfloat16()
                out = woq_model(data).bfloat16()
                torch.testing.assert_close(out_ref, out, atol=1.5e-4, rtol=1.6e-2)

        shape_list = [
            [196, 4096, 4096],
            [1024, 512, 512],
        ]
        use_bias_list = [True, False]
        w_dtype_list = [WoqWeightDtype.INT8, WoqWeightDtype.INT4, WoqWeightDtype.NF4]
        lowp_mode_list = [WoqLowpMode.BF16, WoqLowpMode.INT8]
        group_size_list = [-1, 128]
        cases = itertools.product(
            shape_list, use_bias_list, w_dtype_list, lowp_mode_list, group_size_list
        )
        for shape, use_bias, w_dtype, lowp_mode, group_size in cases:
            test(shape, use_bias, w_dtype, lowp_mode, group_size)

    def test_weight_only_quantization_int8_lowp_mode_int8(self):
        class Mod(nn.Module):
            def __init__(self, has_bias):
                super(Mod, self).__init__()
                self.linear = torch.nn.Linear(K, N, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(has_bias, act_quant_mode, M, group_size):
            dtype = torch.float
            model = Mod(has_bias)
            m = model.eval()
            # m.linear.weight.data = torch.ones_like(m.linear.weight.data)
            m2 = copy.deepcopy(m)
            data = torch.rand(M, K) * 0.5
            qconfig_mapping = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=WoqWeightDtype.INT8,
                lowp_mode=WoqLowpMode.INT8,
                act_quant_mode=act_quant_mode,
                group_size=group_size,
            )
            is_act_sym_quant = act_quant_mode in [4, 5, 6, 7]
            act_quant_group = group_size if group_size > 0 else 64
            fake_quant_x = (
                self._fakequant_by_group_sym(data, act_quant_mode, act_quant_group)
                if is_act_sym_quant
                else self._fakequant_by_group(data, act_quant_mode, act_quant_group)
            )
            prepared_model = prepare(m2, qconfig_mapping, inplace=True)
            with torch.no_grad(), torch.autocast(
                device_type="cpu", enabled=True, dtype=dtype
            ):
                woq_model = convert(prepared_model)
                w_scales = woq_model.linear._op_context.get_scales()
                w_zero_points = woq_model.linear._op_context.get_zero_points()
                w = copy.deepcopy(m.linear.weight.data)
                if group_size > 0:
                    qw, _, _ = quantize_per_block(
                        w,
                        WoqWeightDtype.INT8,
                        group_size,
                        w_scales,
                        w_zero_points,
                        sym_quant=True,
                    )
                else:
                    qw, _, _ = quantize_per_channel(
                        w, WoqWeightDtype.INT8, w_scales, w_zero_points, sym_quant=True
                    )
                if group_size > 0:
                    fake_quant_w = dequantize_per_block(
                        qw,
                        w_scales,
                        w_zero_points,
                        WoqWeightDtype.INT8,
                        group_size,
                        weight_shape=w.shape,
                    )
                else:
                    fake_quant_w = dequantize_per_channel(
                        qw, w_scales, w_zero_points, WoqWeightDtype.INT8, w.shape
                    )
                m.linear.weight.data = fake_quant_w
                y_ref = m(fake_quant_x).to(dtype)
                y = woq_model(data)
                try:
                    torch.testing.assert_close(y, y_ref, atol=1e-2 * 5, rtol=1e-1 * 2)
                except Exception:
                    # The fallback kernel does not support act quant mode
                    # It computes in fp32 by dequantizing weight.
                    fake_quant_w = qw.dequantize()
                    y_ref = data @ fake_quant_w.T + (m.linear.bias if has_bias else 0)
                    y_ref = y_ref.to(dtype)
                    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-1)

        N, K = 64, 512
        has_bias_list = [False, True]
        quant_mode_list = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size_list = [4, 1024]
        group_size_list = [-1, 128]
        cases = itertools.product(
            has_bias_list, quant_mode_list, batch_size_list, group_size_list
        )
        for has_bias, quant_mode, M, group_size in cases:
            test(has_bias, quant_mode, M, group_size)

    def test_weight_only_quantization_sym_quant_weight(self):

        class Mod(nn.Module):
            def __init__(self, has_bias):
                super(Mod, self).__init__()
                self.linear = torch.nn.Linear(K, N, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(
            weight_dtype,
            lowp_mode,
            group_size,
            has_bias,
            act_quant_mode,
            M,
            enable_autocast,
        ):
            amp_dtype = torch.bfloat16
            model = Mod(has_bias)
            m = model.eval()
            m2 = copy.deepcopy(m)
            data = torch.rand(M, K) * 0.5
            qconfig_mapping = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=weight_dtype,
                lowp_mode=lowp_mode,
                act_quant_mode=act_quant_mode,
                group_size=group_size,
                weight_qscheme=WoqWeightQScheme.SYMMETRIC,
            )
            x = data
            prepared_model = prepare(m2, qconfig_mapping, inplace=True)
            with torch.no_grad(), torch.autocast(
                device_type="cpu",
                enabled=enable_autocast,
                dtype=amp_dtype if enable_autocast else None,
            ):
                woq_model = convert(prepared_model)
                # Behavior of WOQ Linear to simulate:
                # Quantize weight to int4 by float qparams at quantization time
                # Quantize activation to int8 at runtime
                # Convert weight and its zero points to INT8 for computation
                packed_weight = woq_model.linear._op_context.get_weight()
                qw = woq_model.linear._op_context.to_public(packed_weight)
                w_scales = woq_model.linear._op_context.get_scales()
                w_zero_points = woq_model.linear._op_context.get_zero_points()
                w = copy.deepcopy(m.linear.weight.data)
                if group_size <= 0:
                    qw, _, _ = quantize_per_channel(
                        w, weight_dtype, w_scales, w_zero_points, sym_quant=True
                    )
                    fake_quant_w = dequantize_per_channel(
                        qw, w_scales, w_zero_points, weight_dtype, w.shape
                    )
                else:
                    qw, _, _ = quantize_per_block(
                        w,
                        weight_dtype,
                        group_size,
                        w_scales,
                        w_zero_points,
                        sym_quant=True,
                    )
                    fake_quant_w = dequantize_per_block(
                        qw, w_scales, w_zero_points, weight_dtype, group_size, w.shape
                    )
                m.linear.weight.data = fake_quant_w
                if lowp_mode == WoqLowpMode.INT8:
                    if packed_weight.dim() == 4:
                        block_k = packed_weight.size(2)
                        groupsize = group_size if group_size > 0 else block_k
                        fake_quant_x = (
                            self._fakequant_by_group(data, act_quant_mode, groupsize)
                            if act_quant_mode < 4
                            else self._fakequant_by_group_sym(
                                data, act_quant_mode, groupsize
                            )
                        )
                        x = fake_quant_x
                y_ref = m(x)
                y = woq_model(data)
                try:
                    torch.testing.assert_close(y, y_ref, atol=1e-2 * 5, rtol=1e-1 * 2)
                except Exception:
                    # The fallback kernel does not support act quant mode
                    # It computes in fp32 by dequantizing weight.
                    fake_quant_w = qw.dequantize()
                    y_ref = data @ fake_quant_w.T + (m.linear.bias if has_bias else 0)
                    y_ref = y_ref
                    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-1)

        N, K = 64, 256
        weight_dtype_list = [WoqWeightDtype.INT8, WoqWeightDtype.INT4]
        lowp_mode_list = [WoqLowpMode.FP16, WoqLowpMode.BF16, WoqLowpMode.INT8]
        group_size_list = [-1, 64]
        has_bias_list = [False, True]
        quant_a_mode_list = [
            WoqActQuantMode.PER_TENSOR,
            WoqActQuantMode.PER_IC_BLOCK,
            WoqActQuantMode.PER_BATCH,
            WoqActQuantMode.PER_BATCH_IC_BLOCK,
            WoqActQuantMode.PER_TENSOR_SYM,
            WoqActQuantMode.PER_IC_BLOCK_SYM,
            WoqActQuantMode.PER_BATCH_SYM,
            WoqActQuantMode.PER_BATCH_IC_BLOCK_SYM,
        ]
        batch_size_list = [4, 1024]
        enable_autocast_list = [False, True]
        cases = itertools.product(
            weight_dtype_list,
            lowp_mode_list,
            group_size_list,
            has_bias_list,
            quant_a_mode_list,
            batch_size_list,
            enable_autocast_list,
        )
        for (
            weight_dtype,
            lowp_mode,
            group_size,
            has_bias,
            quant_a_mode,
            M,
            enable_autocast,
        ) in cases:
            test(
                weight_dtype,
                lowp_mode,
                group_size,
                has_bias,
                quant_a_mode,
                M,
                enable_autocast,
            )

    def test_weight_only_quantization_without_op_context(self):
        class M(nn.Module):
            def __init__(self, input_channel, output_channel, has_bias):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(feature, has_bias, w_dtype, lowp_mode, enable_amp):
            if w_dtype == WoqWeightDtype.NF4 and lowp_mode == WoqLowpMode.INT8:
                return
            model = M(feature[1], feature[2], has_bias)
            m = model.eval()
            data = torch.rand(feature[0], feature[1])
            weight = model.linear.weight.clone()
            bias = model.linear.bias
            weight_shape = weight.shape
            group_size = 128
            sym_quant = (
                w_dtype == WoqWeightDtype.INT8 and lowp_mode == WoqLowpMode.INT8
            ) or w_dtype == WoqWeightDtype.NF4
            qweight, w_scales, w_zero_points = quantize_per_block(
                weight, w_dtype, group_size, sym_quant=sym_quant
            )
            dtype_to_str = {
                WoqWeightDtype.INT8: "int8",
                WoqWeightDtype.INT4: "int4",
                WoqWeightDtype.NF4: "nf4",
            }
            packed_weight, new_scales, new_zeros, new_bias, compensation = (
                torch.ops.ipex_prepack.woq_linear_pack_weight(
                    qweight,
                    dtype_to_str[w_dtype],
                    weight_shape,
                    w_scales,
                    w_zero_points,
                    bias,
                    None,
                    group_size,
                    lowp_mode,
                )
            )
            unpacked_weight = torch.ops.ipex_prepack.woq_linear_unpack_weight(
                packed_weight,
                dtype_to_str[w_dtype],
                weight_shape,
                lowp_mode,
            )
            assert torch.equal(unpacked_weight, qweight)

            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=w_dtype,
                lowp_mode=lowp_mode,
                group_size=group_size,
            )
            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)
            with torch.no_grad(), torch.autocast(
                device_type="cpu", enabled=enable_amp, dtype=torch.bfloat16
            ):
                woq_model = convert(prepared_model)
                output_ref = woq_model(data)
                output = torch.ops.torch_ipex.woq_linear(
                    data,
                    packed_weight,
                    dtype_to_str[w_dtype],
                    weight_shape,
                    new_scales,
                    new_zeros,
                    new_bias,
                    None,
                    group_size,
                    lowp_mode,
                    WoqActQuantMode.PER_BATCH_IC_BLOCK_SYM,
                    compensation,
                )
                torch.testing.assert_close(output, output_ref)

        shape_list = [
            [4, 1024, 1024],
            [1024, 512, 512],
            [4, 256, 272],
        ]
        use_bias_list = [True, False]
        w_dtype_list = [WoqWeightDtype.INT8, WoqWeightDtype.INT4, WoqWeightDtype.NF4]
        lowp_mode_list = lowp_mode_list = [
            WoqLowpMode.NONE,
            WoqLowpMode.FP16,
            WoqLowpMode.BF16,
            WoqLowpMode.INT8,
        ]
        enable_amp_list = [True, False]
        cases = itertools.product(
            shape_list, use_bias_list, w_dtype_list, lowp_mode_list, enable_amp_list
        )
        for shape, use_bias, w_dtype, lowp_mode, enable_amp in cases:
            test(shape, use_bias, w_dtype, lowp_mode, enable_amp)

    def test_weight_padding(self):
        """
        If N of weight shape N * K is not a multiple of block_n, it is padded to be a multiple of block_n.
        """

        class Mod(nn.Module):
            def __init__(self, input_channel, output_channel, has_bias):
                super(Mod, self).__init__()
                self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(M, has_bias, w_dtype):
            N, K, N_padded = 500, 512, 512
            model = Mod(K, N, has_bias)
            m = model.eval()
            m_ref = Mod(K, N_padded, False).eval()
            data = torch.rand(M, K)
            weight = model.linear.weight
            weight_int4, w_scales, w_zero_points = quantize_per_channel(
                weight,
                w_dtype,
                sym_quant=True if w_dtype == WoqWeightDtype.NF4 else False,
            )
            weight_fp32 = dequantize_per_channel(
                weight_int4, w_scales, w_zero_points, w_dtype, weight.shape
            )
            if has_bias:
                bias = model.linear.bias
                output1 = torch.matmul(data, weight_fp32.T) + bias
            else:
                output1 = torch.matmul(data, weight_fp32.T)

            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=w_dtype
            )
            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)
            prepared_model_ref = prepare(
                m_ref, qconfig, example_inputs=data, inplace=False
            )
            with torch.no_grad():
                woq_model = convert(prepared_model)
                woq_model_ref = convert(prepared_model_ref)
                assert (
                    woq_model.linear.weight.shape == woq_model_ref.linear.weight.shape
                )

                output2 = woq_model(data)
                torch.testing.assert_close(output1, output2)

        M_list = [4, 1024]
        use_bias_list = [True, False]
        w_dtype_list = [WoqWeightDtype.INT8, WoqWeightDtype.INT4, WoqWeightDtype.NF4]
        cases = itertools.product(M_list, use_bias_list, w_dtype_list)
        for M, use_bias, w_dtype in cases:
            test(M, use_bias, w_dtype)

    def test_weight_only_quantization_nf4_lowp_mode_int8(self):

        class Mod(nn.Module):
            def __init__(self, has_bias):
                super(Mod, self).__init__()
                self.linear = torch.nn.Linear(K, N, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(has_bias, act_quant_mode, M):
            dtype = torch.bfloat16
            model = Mod(has_bias)
            m = model.eval()
            m2 = copy.deepcopy(m)
            data = torch.rand(M, K) * 0.5
            qconfig_mapping = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=WoqWeightDtype.NF4,
                lowp_mode=WoqLowpMode.INT8,
                act_quant_mode=act_quant_mode,
                group_size=groupsize,
            )
            fake_quant_x = (
                self._fakequant_by_group(data, act_quant_mode, groupsize)
                if act_quant_mode < 4
                else self._fakequant_by_group_sym(data, act_quant_mode, groupsize)
            )
            prepared_model = prepare(m2, qconfig_mapping, inplace=True)
            with torch.no_grad(), torch.autocast(
                device_type="cpu", enabled=True, dtype=dtype
            ):
                woq_model = convert(prepared_model)
                w_scales = woq_model.linear._op_context.get_scales()
                w_zero_points = woq_model.linear._op_context.get_zero_points()
                w = copy.deepcopy(m.linear.weight.data)
                qw, _, _ = quantize_per_block(
                    w,
                    WoqWeightDtype.NF4,
                    groupsize,
                    w_scales,
                    w_zero_points,
                    sym_quant=True,
                )
                fake_quant_w = dequantize_per_block(
                    qw,
                    w_scales,
                    w_zero_points,
                    WoqWeightDtype.NF4,
                    groupsize,
                    w.shape,
                    dequant_nf4_via_int8=True,
                )
                m.linear.weight.data = fake_quant_w
                y_ref = m(fake_quant_x).to(dtype)
                y = woq_model(data)
                try:
                    torch.testing.assert_close(y, y_ref, atol=1e-2 * 5, rtol=1e-1 * 2)
                except Exception:
                    # The fallback kernel does not support act quant mode
                    # It computes in fp32 by dequantizing weight.
                    fake_quant_w = dequantize_per_block(
                        qw,
                        w_scales,
                        w_zero_points,
                        WoqWeightDtype.NF4,
                        groupsize,
                        w.shape,
                    )
                    y_ref = data @ fake_quant_w.T + (m.linear.bias if has_bias else 0)
                    y_ref = y_ref.to(dtype)
                    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-1)

        N, K = 64, 512
        groupsize = 64
        has_bias_list = [False, True]
        quant_mode_list = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size_list = [4, 1024]
        cases = itertools.product(has_bias_list, quant_mode_list, batch_size_list)
        for has_bias, quant_mode, M in cases:
            test(has_bias, quant_mode, M)

    def test_pack_gptq_weight(self):
        from intel_extension_for_pytorch.utils.weight_only_quantization import (
            _convert_optimum_format_to_desired,
            _convert_gptq_scales_qzeros,
        )

        N, K = 64, 256
        G = 128  # group size
        int32_min = -(2**31)
        int32_max = 2**31 - 1
        # gptq weight shape = [K // 8, N]
        # gptq scales shape = [K // G, N]
        # gptq qzeros shape = [K // G, N // 8]
        qweight_gptq = torch.randint(
            int32_min, int32_max, (K // 8, N), dtype=torch.int32
        )
        scales = torch.rand((K // G, N), dtype=torch.float16)
        qzeros = torch.randint(
            int32_min, int32_max, (K // G, N // 8), dtype=torch.int32
        )
        new_scales, new_qzeros = _convert_gptq_scales_qzeros(scales, qzeros, False)
        new_qweight, new_scales_ref, new_qzeros_ref = (
            _convert_optimum_format_to_desired(qweight_gptq, scales, qzeros, False)
        )
        for lowp_mode in [WoqLowpMode.BF16, WoqLowpMode.INT8]:
            # We only support g_idx is None
            op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack_int4(
                qweight_gptq,
                new_scales,
                new_qzeros,
                None,  # bias
                None,  # g_idx
                None,  # batch size
                G,
                int(lowp_mode),
                WoqActQuantMode.PER_BATCH_IC_BLOCK_SYM,
                False,  # cache weight
                WoqWeightFormat.GPTQ_FORMAT,
            )
            packed_weight = op_context.get_weight()
            op_context_ref = torch.ops.ipex_prepack.weight_only_qlinear_prepack_int4(
                new_qweight,
                new_scales_ref,
                new_qzeros_ref,
                None,  # bias
                None,  # g_idx
                None,  # batch size
                G,
                int(lowp_mode),
                WoqActQuantMode.PER_BATCH_IC_BLOCK_SYM,
                False,  # cache weight
                WoqWeightFormat.PLAIN_FORMAT,
            )
            packed_weight_ref = op_context_ref.get_weight()

            torch.testing.assert_close(packed_weight, packed_weight_ref)

    def test_pack_awq_weight(self):
        from intel_extension_for_pytorch.nn.utils._model_convert import (
            prepack_awq_weight,
            _convert_awq_scales_qzeros,
        )

        N, K = 64, 256
        G = 128  # group size
        int32_min = -(2**31)
        int32_max = 2**31 - 1
        # awq weight shape = [K, N // 8]
        # awq scales shape = [K // G, N]
        # awq qzeros shape = [K // G, N // 8]
        qweight_awq = torch.randint(
            int32_min, int32_max, (K, N // 8), dtype=torch.int32
        )
        scales = torch.rand((K // G, N), dtype=torch.float16)
        qzeros = torch.randint(
            int32_min, int32_max, (K // G, N // 8), dtype=torch.int32
        )
        new_scales, new_qzeros = _convert_awq_scales_qzeros(scales, qzeros)
        new_qweight, new_scales_ref, new_qzeros_ref = prepack_awq_weight(
            qweight_awq, qzeros, scales, bits=4, group_size=G
        )
        for lowp_mode in [WoqLowpMode.BF16, WoqLowpMode.INT8]:
            op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack_int4(
                qweight_awq,
                new_scales,
                new_qzeros,
                None,  # bias
                None,  # g_idx
                None,  # batch size
                G,
                int(lowp_mode),
                WoqActQuantMode.PER_BATCH_IC_BLOCK_SYM,
                False,  # cache weight
                WoqWeightFormat.AWQ_FORMAT,
            )
            packed_weight = op_context.get_weight()
            op_context_ref = torch.ops.ipex_prepack.weight_only_qlinear_prepack_int4(
                new_qweight,
                new_scales_ref,
                new_qzeros_ref,
                None,  # bias
                None,  # g_idx
                None,  # batch size
                G,
                int(lowp_mode),
                WoqActQuantMode.PER_BATCH_IC_BLOCK_SYM,
                False,  # cache weight
                WoqWeightFormat.PLAIN_FORMAT,
            )
            packed_weight_ref = op_context_ref.get_weight()

            torch.testing.assert_close(packed_weight, packed_weight_ref)

    def test_fp8_weight(self):
        class Mod(nn.Module):
            def __init__(self, input_channel, output_channel, has_bias):
                super(Mod, self).__init__()
                self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(feature, has_bias, dtype):
            M, K, N = feature[0], feature[1], feature[2]
            model = Mod(K, N, has_bias)
            m = model.eval()
            data = torch.rand(M, K).to(dtype) * 0.1
            weight = model.linear.weight
            group_size = 128
            fp8_max = 448.0
            grouped_shape = (N, K // group_size, group_size)
            weight_grouped = weight.view(grouped_shape)
            w_scales = weight_grouped.abs().max(-1)[0] / fp8_max
            w_zero_points = None
            w_fp8 = weight_grouped / w_scales.unsqueeze(-1)
            w_fp8 = w_fp8.view(N, K).to(torch.float8_e4m3fn)
            w_dq_bf16 = w_fp8.view(grouped_shape).float() * w_scales.unsqueeze(-1)
            compute_dtype = dtype
            if dtype == torch.bfloat16 and M <= 4:
                compute_dtype = torch.half
            lowp_mode = (
                WoqLowpMode.BF16 if dtype == torch.bfloat16 else WoqLowpMode.NONE
            )

            w_dq_bf16 = w_dq_bf16.view(N, K).to(compute_dtype)

            if has_bias:
                bias = model.linear.bias
                output1 = torch.matmul(data.to(compute_dtype), w_dq_bf16.T) + bias
            else:
                output1 = torch.matmul(data.to(compute_dtype), w_dq_bf16.T).float()

            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=WoqWeightDtype.FP8,
                lowp_mode=lowp_mode,
            )
            with torch.no_grad():
                woq_m = copy.deepcopy(m)
                m.linear.qconfig = qconfig.global_qconfig
                woq_m.linear = WeightOnlyQuantizedLinear.from_float_and_qweight(
                    m.linear,
                    w_fp8,
                    WoqWeightDtype.FP8,
                    w_scales,
                    w_zero_points,
                    m.linear.bias,
                    group_size,
                )
                assert (
                    woq_m.linear.weight is not None
                    and woq_m.linear.weight.dtype == torch.float8_e4m3fn
                )

                output2 = woq_m(data).float()
                torch.testing.assert_close(output1, output2, atol=1e-3, rtol=1e-3)

        shape_list = [
            [4, 1024, 1024],
            [1024, 1024, 1024],
        ]
        use_bias_list = [True, False]
        dtype_list = [torch.bfloat16, torch.float]
        cases = itertools.product(shape_list, use_bias_list, dtype_list)
        for shape, use_bias, dtype in cases:
            test(shape, use_bias, dtype)


class QuantizedOpTester(TestCase):
    def test_dequantize_nf4(self):
        dtype_list = [torch.float, torch.bfloat16, torch.half]
        group_size_list = [-1, 32, 128]
        cases = itertools.product(dtype_list, group_size_list)
        for dtype, group_size in cases:
            t_fp = torch.randn(1024, 1024, dtype=dtype)
            scale_dtype_list = list(set([dtype, torch.float]))
            for scale_dtype in scale_dtype_list:
                if group_size < 0:
                    t, scales, zp = quantize_per_channel(
                        t_fp, WoqWeightDtype.NF4, None, None, sym_quant=True
                    )
                    scales = scales.to(scale_dtype)
                    out_ref = dequantize_per_channel(
                        t, scales, zp, WoqWeightDtype.NF4, t_fp.shape
                    ).to(dtype)
                    out = torch.ops.torch_ipex.dequantize_nf4(
                        t, scales, group_size, dtype
                    )
                    assert torch.allclose(out, out_ref)
                else:
                    t, scales, zp = quantize_per_block(
                        t_fp, WoqWeightDtype.NF4, group_size, None, None, sym_quant=True
                    )
                    scales = scales.to(scale_dtype)
                    out_ref = dequantize_per_block(
                        t, scales, zp, WoqWeightDtype.NF4, group_size, t_fp.shape
                    ).to(dtype)
                    out = torch.ops.torch_ipex.dequantize_nf4(
                        t, scales, group_size, dtype
                    )
                    assert torch.allclose(out, out_ref)


if __name__ == "__main__":
    test = unittest.main()
    run_tests()
