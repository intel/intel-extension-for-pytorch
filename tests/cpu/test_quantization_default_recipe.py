import itertools
import tempfile
import torch
import torch.nn as nn
from torch.testing import FileCheck
from torch.ao.quantization import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
    QConfig,
    QConfigMapping,
)
import copy

import intel_extension_for_pytorch as ipex
from test_ao_jit_llga_utils import JitLlgaTestCase, LLGA_FUSION_GROUP
from torch.testing._internal.common_utils import run_tests
from torch.ao.nn.quantized.modules.utils import _quantize_weight
from intel_extension_for_pytorch.quantization import prepare, convert


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
        for quantized_modules, pattern in zip(quantized_modules, patterns):
            m = M(quantized_modules).eval()

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

    def test_smooth_quant(self):
        N, IC, OC = 4, 4, 4
        x_data = [(i + 1) ** 3 for i in range(N)]
        x = torch.Tensor(x_data).repeat(N, 1)
        w_data = [(i + 1) for i in range(N)]
        w = torch.Tensor(w_data).repeat(OC, 1)

        class Mod(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dense = nn.Linear(IC, OC)
                self.dense.weight = nn.Parameter(w)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.dense(x)
                x = self.relu(x)
                return x

        for bf16_mixed in [False, True]:
            with torch.no_grad(), torch.autocast(
                device_type="cpu", enabled=bf16_mixed, dtype=torch.bfloat16
            ):
                m = Mod().eval()
                alpha = 0.5
                qconfig_mapping = ipex.quantization.get_smooth_quant_qconfig_mapping(
                    alpha=alpha
                )
                prepared_model = ipex.quantization.prepare(
                    copy.deepcopy(m), qconfig_mapping, example_inputs=x, inplace=False
                )
                prepared_model(x)
                converted_model = ipex.quantization.convert(prepared_model)
                traced_model = torch.jit.trace(converted_model, x)
                traced_model = torch.jit.freeze(traced_model)
                # Check graph
                # Do not run traced_model to fuse by LLGA because `mul`
                # may be fused to LLGA fusion group and cannot be found by the following code
                graph = traced_model.graph_for(x)
                found_mul = False
                for node in graph.nodes():
                    if node.kind() == "aten::mul":
                        found_mul = True
                assert (
                    found_mul
                ), "Failed to find the inserted `mul` before Linear for SmoothQuant"
                traced_model(x)
                result_sq = traced_model(x)

                # Check correctness with reference quantized model
                # Calculate and apply scaling factors manually to model and use default static quant
                x_max_per_ic = torch.max(x, 0)[0]
                w_max_per_ic = torch.max(w, 0)[0]
                act_scaling_factors = torch.pow(w_max_per_ic, 1 - alpha) / torch.pow(
                    x_max_per_ic, alpha
                )
                wei_scaling_factors = torch.pow(x_max_per_ic, alpha) / torch.pow(
                    w_max_per_ic, 1 - alpha
                )
                new_x = torch.mul(x, act_scaling_factors)
                new_w = torch.mul(w, wei_scaling_factors)
                m2 = copy.deepcopy(m)
                m2.dense.weight = nn.Parameter(new_w)
                # SmoothQuant uses MinMaxObserver for activation not histogram observer
                w_observer = PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8, qscheme=torch.per_channel_symmetric
                )
                static_qconfig = QConfig(
                    activation=MinMaxObserver.with_args(reduce_range=False),
                    weight=w_observer,
                )
                qconfig_mapping = QConfigMapping().set_global(static_qconfig)
                prepared_model2 = ipex.quantization.prepare(
                    m2, qconfig_mapping, example_inputs=new_x, inplace=False
                )
                prepared_model2(new_x)
                converted_model2 = ipex.quantization.convert(prepared_model2)
                traced_model2 = torch.jit.trace(converted_model2, new_x)
                traced_model2 = torch.jit.freeze(traced_model2)
                traced_model2(new_x)
                traced_model2(new_x)
                result_ref = traced_model2(new_x)
                assert torch.allclose(result_sq, result_ref)

    def test_smooth_quant_save_load_qconf_summary(self):
        class Mod(nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = nn.Linear(4, 4)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.dense(x))

        m = Mod().eval()
        x = torch.rand(1, 4)
        calib_dataset = [torch.rand(1, 4) for _ in range(5)]
        per_channel_observer = (
            torch.ao.quantization.MovingAveragePerChannelMinMaxObserver
        )
        custom_config = {
            "alpha": 0.75,
            "act_observer": torch.ao.quantization.MinMaxObserver(),
            "act_ic_observer": per_channel_observer(ch_axis=-1),
            "wei_observer": per_channel_observer(
                dtype=torch.qint8, qscheme=torch.per_channel_symmetric
            ),
            "wei_ic_observer": per_channel_observer(ch_axis=1),
        }
        for use_custom_config in [False, True]:
            kwargs = custom_config if use_custom_config else {}
            qconfig_mapping = ipex.quantization.get_smooth_quant_qconfig_mapping(
                **kwargs
            )
            prepared_model = ipex.quantization.prepare(
                m, qconfig_mapping, example_inputs=x, inplace=False
            )

            # Save observer info for comparison
            if use_custom_config:
                observer_info = {
                    **prepared_model._fqn_to_auto_quant_state_map[
                        " "
                    ].tensor_id_to_observer,
                    **prepared_model._fqn_to_auto_quant_state_map[
                        " "
                    ].weight_tensor_id_to_observer,
                }
                observer_info_dict = {}
                for key, obs in observer_info.items():
                    observer_info_dict[key] = {
                        "smooth_quant_enabled": obs.smooth_quant_enabled,
                        "alpha": obs.alpha,
                        "ic_obs": type(obs.ic_obs),
                        "act_obs": type(obs.act_obs),
                    }

            for data in calib_dataset:
                prepared_model(data)

            with tempfile.NamedTemporaryFile() as fp:
                qconf_filename = fp.name
                prepared_model.save_qconf_summary(qconf_summary=qconf_filename)
                q_model = ipex.quantization.convert(prepared_model)

                with torch.no_grad():
                    q_model = torch.jit.trace(q_model, x)
                    q_model = torch.jit.freeze(q_model)
                out_ref = q_model(x)

                prepared_model_2 = ipex.quantization.prepare(
                    m, qconfig_mapping, example_inputs=x, inplace=False
                )
                prepared_model_2.load_qconf_summary(qconf_summary=qconf_filename)

                # Save observer info for comparison
                if use_custom_config:
                    observer_info_2 = {
                        **prepared_model_2._fqn_to_auto_quant_state_map[
                            " "
                        ].tensor_id_to_observer,
                        **prepared_model_2._fqn_to_auto_quant_state_map[
                            " "
                        ].weight_tensor_id_to_observer,
                    }
                    observer_info_dict_2 = {}
                    for key, obs in observer_info_2.items():
                        observer_info_dict_2[key] = {
                            "smooth_quant_enabled": obs.smooth_quant_enabled,
                            "alpha": obs.alpha,
                            "ic_obs": type(obs.ic_obs),
                            "act_obs": type(obs.act_obs),
                        }

                q_model_2 = ipex.quantization.convert(prepared_model_2)

                with torch.no_grad():
                    q_model_2 = torch.jit.trace(q_model_2, x)
                    q_model_2 = torch.jit.freeze(q_model_2)
                out_2 = q_model_2(x)

                assert torch.allclose(out_ref, out_2)

            # Check observers
            if use_custom_config:
                assert (
                    observer_info_dict == observer_info_dict_2
                ), "Error: SmoothQuant observer info lost after saving/loading qconf JSON"

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
            weight_observer = (
                ipex.quantization.get_weight_only_quant_qconfig_mapping().global_qconfig.weight()
            )
            weight_observer(weight)
            weight_int8 = _quantize_weight(weight, weight_observer)
            weight_fp32 = weight_int8.dequantize()
            if has_bias:
                bias = model.linear.bias
                output1 = torch.matmul(data, weight_fp32.T) + bias
            else:
                output1 = torch.matmul(data, weight_fp32.T)

            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping()
            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)
            with torch.no_grad():
                woq_model = convert(prepared_model)
                woq_linear_class = (
                    ipex.nn.modules.weight_only_quantization.IpexWoqLinear
                )
                assert isinstance(woq_model.linear, woq_linear_class)

                output2 = woq_model(data)
                torch.testing.assert_close(output1, output2)

        shape_list = [
            [3, 31, 31],
            [4, 4096, 4096],
            [9, 4095, 4095],
            [9, 4096, 4096],
            [196, 4095, 16383],
            [192, 4096, 16384],
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

        def tpp_is_used(N, K):
            num_threads = torch.get_num_threads()
            block_n = 32 if N // 64 // num_threads < 4 else 64
            block_k = 64
            while K % block_k != 0:
                block_k //= 2
                assert block_k > 0
            return N % block_n == 0 and K % block_k == 0

        def test(feature, has_bias, w_dtype):
            model = M(feature[1], feature[2], has_bias)
            m = model.eval()
            data = torch.rand(feature[0], feature[1])
                
            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(weight_dtype=w_dtype)
            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)
            
            with torch.no_grad():
                weight = m.linear.weight
                weight_observer = qconfig.global_qconfig.weight()
                weight_observer(weight)
                weight_int8 = _quantize_weight(weight, weight_observer)
                weight_fp32 = weight_int8.dequantize()
                weight_bf16 = weight_fp32.bfloat16()
                weight_fp16 = weight_fp32.half()
                data_bf16 = data.bfloat16()
                data_fp16 = data_bf16.half()
                bias_fp32 = m.linear.bias
                use_tpp = tpp_is_used(feature[2], feature[1])
                if use_tpp:
                    # if M >= 32, compute in bf16
                    # if M < 32, compute in fp32 or fp16. Depends on fp16 support.
                    if feature[0] >= 32:
                        output1 = torch.matmul(data_bf16.float(), weight_bf16.float().T).bfloat16()
                        if has_bias:
                            output1 = output1 + bias_fp32.bfloat16()
                    else:
                        output1_fp32 = torch.matmul(data_bf16.float(), weight_bf16.float().T)
                        if has_bias:
                            output1_fp32 = output1_fp32 + bias_fp32
                        output1_fp16 = torch.matmul(data_fp16.float(), weight_fp16.float().T).half()
                        if has_bias:
                            output1_fp16 = output1_fp16 + bias_fp32.half()
                else:
                    if feature[0] <= 4:
                        output1 = torch.matmul(data_bf16.float(), weight_fp32.T)
                    else:
                        output1 = torch.matmul(data_bf16.float(), weight_bf16.float().T)
                    if has_bias:
                        output1 = output1 + bias_fp32
                    output1 = output1.bfloat16()
                with torch.autocast(device_type='cpu', enabled=True, dtype= torch.bfloat16):
                    woq_model = convert(prepared_model)
                    woq_linear_class = ipex.nn.modules.weight_only_quantization.IpexWoqLinear
                    assert isinstance(woq_model.linear, woq_linear_class)   
                                            
                    woq_model = torch.jit.trace(woq_model, data)
                    woq_model = torch.jit.freeze(woq_model)
                    output2 = woq_model(data)
                    output2 = output2.bfloat16()
                if use_tpp and feature[0] < 32:
                    try:
                        torch.testing.assert_close(output1_fp32.bfloat16(), output2, atol=0.01, rtol=0.1)
                    except Exception as e:
                        torch.testing.assert_close(output1_fp16.bfloat16(), output2, atol=0.01, rtol=0.1)
                else:
                    torch.testing.assert_close(output1, output2)

        shape_list = [
            [3, 31, 31],
            # [4, 4096, 4096], # not supported by TPP yet
            [9, 4095, 4095],
            [196, 4095, 16383],
        ]
        use_bias_list = [True, False]
        w_dtype_list = [torch.qint8]
        cases = itertools.product(shape_list, use_bias_list, w_dtype_list)
        for shape, use_bias, w_dtype in cases:
            test(shape, use_bias, w_dtype)


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

            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(weight_dtype=w_dtype)
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
            [9, 4095, 4095],
            [196, 4095, 16383],
        ]
        use_bias_list = [True, False]
        w_dtype_list = [torch.qint8]
        cases = itertools.product(shape_list, use_bias_list, w_dtype_list)
        for shape, use_bias, w_dtype in cases:
            test(shape, use_bias, w_dtype)


    def test_weight_only_quantization_eltwise_fusion(self):
        return # disabled for now
        class M(nn.Module):
            def __init__(self, input_channel, output_channel, has_bias, post_op):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)
                self.post_op = post_op

            def forward(self, x):
                return self.post_op(self.linear(x))

        def test(feature, has_bias, post_op):
            post_op_name, post_op = post_op
            model = M(feature[1], feature[2], has_bias, post_op)
            m = model.eval()
            data = torch.rand(feature[0], feature[1])
            weight = model.linear.weight
            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping()
            weight_observer = qconfig.global_qconfig.weight()
            weight_observer(weight)
            weight_int8 = _quantize_weight(weight, weight_observer)
            weight_fp32 = weight_int8.dequantize()
            if has_bias:
                bias = model.linear.bias
                output1 = torch.matmul(data, weight_fp32.T) + bias
            else:
                output1 = torch.matmul(data, weight_fp32.T)
            output1 = post_op(output1)

            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)
            with torch.no_grad():
                woq_model = convert(prepared_model)
                woq_linear_class = (
                    ipex.nn.modules.weight_only_quantization.IpexWoqLinear
                )
                assert isinstance(woq_model.linear, woq_linear_class)
                woq_model = torch.jit.trace(woq_model, data)
                woq_model = torch.jit.freeze(woq_model)
                woq_model(data)
                node_kind_list = [node.kind() for node in woq_model.graph_for(data).nodes()]
                linear_node_kind = 'torch_ipex::ipex_woq_linear'
                fused_node_kind = 'torch_ipex::woq_linear_' + post_op_name + '_run'
                assert linear_node_kind not in node_kind_list and fused_node_kind in node_kind_list

                output2 = woq_model(data)

                torch.testing.assert_close(output1, output2)

        shape_list = [
            [4, 24, 24],
            [4, 64, 64],
            [196, 64, 64],
        ]
        use_bias_list = [True, False]
        post_op_list = [
            ('relu', nn.ReLU()),
            ('relu', nn.functional.relu_),
            ('gelu', nn.GELU(approximate='none')),
            ('gelu', nn.GELU(approximate='tanh')),
            ('gelu', torch._C._nn.gelu_),
        ]
        cases = itertools.product(shape_list, use_bias_list, post_op_list)
        for shape, use_bias, post_op in cases:
            test(shape, use_bias, post_op)

    def test_weight_only_quantization_add_fusion(self):
        return # disabled for now
        class M(nn.Module):
            # on_the_left = True: linear + Y -> Y
            # on_the_left = False: Y + linear -> Y
            def __init__(self, ic, oc, has_bias, alpha, on_the_left, use_relu):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(ic, oc, has_bias)
                self.relu = nn.ReLU()
                self.alpha = alpha
                self.on_the_left = on_the_left
                self.use_relu = use_relu

            def forward(self, x, y):
                x = self.linear(x)
                if self.on_the_left:
                    y = torch.add(x, y, alpha=self.alpha)
                else:
                    y = torch.add(y, x, alpha=self.alpha)
                if self.use_relu:
                    return self.relu(y)
                return y

        def test(feature, has_bias, alpha, on_the_left, use_relu):
            if not on_the_left and alpha != 1.0:
                # alpha must be 1 in this case
                return
            model = M(feature[1], feature[2], has_bias, alpha, on_the_left, use_relu)
            m = model.eval()
            data = torch.rand(feature[0], feature[1])
            accu = torch.rand(feature[0], feature[1])
            weight = model.linear.weight
            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping()
            weight_observer = qconfig.global_qconfig.weight()
            weight_observer(weight)
            weight_int8 = _quantize_weight(weight, weight_observer)
            weight_fp32 = weight_int8.dequantize()
            if has_bias:
                bias = model.linear.bias
                output1 = torch.matmul(data, weight_fp32.T) + bias
            else:
                output1 = torch.matmul(data, weight_fp32.T)
            if on_the_left:
                output1 = output1 + accu * alpha
            else:
                output1 = accu + output1 * alpha
            if use_relu:
                output1 = nn.functional.relu(output1)

            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)
            with torch.no_grad():
                inputs = (data, copy.deepcopy(accu)) # accu is overwritten
                woq_model = convert(prepared_model)
                woq_model = torch.jit.trace(woq_model, inputs)
                woq_model = torch.jit.freeze(woq_model)
                woq_model(*inputs)
                node_kind_list = [node.kind() for node in woq_model.graph_for(*inputs).nodes()]
                linear_node_kind = 'torch_ipex::ipex_woq_linear'
                fused_node_kind = 'torch_ipex::woq_linear_add_relu_run' if use_relu else 'torch_ipex::woq_linear_add_run'
                assert linear_node_kind not in node_kind_list and fused_node_kind in node_kind_list

                output2 = woq_model(data, accu)

                torch.testing.assert_close(output1, output2)

        shape_list = [
            [4, 24, 24],
            [4, 64, 64],
            [196, 64, 64],
        ]
        use_bias_list = [True, False]
        alpha_list = [1.0, 0.5, 1.5]
        on_the_left_list = [True, False]
        use_relu_list = [True, False]
        cases = itertools.product(shape_list, use_bias_list, alpha_list, on_the_left_list, use_relu_list)
        for shape, use_bias, alpha, on_the_left, use_relu in cases:
            test(shape, use_bias, alpha, on_the_left, use_relu)

    def test_weight_only_quantization_lowp_compute(self):
        from intel_extension_for_pytorch.quantization import WoqLowpMode
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        data = torch.rand(4, 4)
        m = M()
        for mode in [WoqLowpMode.FP16, WoqLowpMode.BF16]:
            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(lowp_mode=mode)
            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)
            with torch.no_grad():
                woq_model = convert(prepared_model)
                assert hasattr(woq_model.linear, '_lowp_mode') and woq_model.linear._lowp_mode == mode, \
                    'Weight-only quantization: low precision gemm flag is not correctly set'

if __name__ == "__main__":
    run_tests()
