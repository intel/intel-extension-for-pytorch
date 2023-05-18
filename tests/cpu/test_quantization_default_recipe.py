import sys
import os
import unittest
import itertools
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import FileCheck
from torch.ao.quantization import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
    QConfig,
    QConfigMapping,
)
import copy
from test_autocast import get_rand_seed

import intel_extension_for_pytorch as ipex
from test_ao_jit_llga_utils import JitLlgaTestCase, run_tests, LLGA_FUSION_GROUP

import intel_extension_for_pytorch as ipex

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
                ["aten::dequantize", "aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
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
            def __init__(self, ):
                super(conv_swish, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 1)

            def forward(self, x):
                x = self.conv(x)
                y = x.sigmoid()
                z = torch.mul(x, y)
                return z

        class conv_eltwise(nn.Module):
            def __init__(self, ):
                super(conv_eltwise, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 1)

            def forward(self, x):
                x = self.conv(x)
                x = x.relu_()
                return x

        # TODO: test more quantized modules(especially for fused module). 
        quantized_modules = [conv_swish(), conv_eltwise()]
        patterns = [
                [["aten::dequantize", "aten::dequantize", "aten::_convolution", "aten::sigmoid", "aten::mul", "aten::quantize_per_tensor"]],
                [["aten::dequantize", "aten::dequantize", "aten::_convolution", "aten::relu", "aten::quantize_per_tensor"]],
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
                ["aten::dequantize", "aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
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
        assert hasattr(converted_model, 'linear')
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

        check_model_obsever_has_run = \
            ipex.quantization._utils.check_model_obsever_has_run
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

        # Use SmoothQuant to quantize the model
        m = Mod().eval()
        alpha = 0.5
        qconfig_mapping = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=alpha)
        prepared_model = ipex.quantization.prepare(
            copy.deepcopy(m), qconfig_mapping, example_inputs=x, inplace=False)
        prepared_model(x)
        converted_model = ipex.quantization.convert(prepared_model)
        with torch.no_grad():
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
        assert found_mul, 'Failed to find the inserted `mul` before Linear for SmoothQuant'
        result_sq = traced_model(x)
        
        # Check correctness with reference quantized model
        # Calculate and apply scaling factors manually to model and use default static quant
        x_max_per_ic = torch.max(x, 0)[0]
        w_max_per_ic = torch.max(w, 0)[0]
        act_scaling_factors = torch.pow(w_max_per_ic, 1 - alpha) / torch.pow(x_max_per_ic, alpha)
        wei_scaling_factors = torch.pow(x_max_per_ic, alpha) / torch.pow(w_max_per_ic, 1 - alpha)
        new_x = torch.mul(x, act_scaling_factors)
        new_w = torch.mul(w, wei_scaling_factors)
        m2 = copy.deepcopy(m)
        m2.dense.weight = nn.Parameter(new_w)
        # SmoothQuant uses MinMaxObserver for activation not histogram observer
        w_observer = PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
        static_qconfig = QConfig(activation=MinMaxObserver.with_args(reduce_range=False),
                                 weight=w_observer)
        qconfig_mapping = QConfigMapping().set_global(static_qconfig)
        prepared_model = ipex.quantization.prepare(m2, qconfig_mapping, example_inputs=new_x, inplace=False)
        prepared_model(new_x)
        converted_model = ipex.quantization.convert(prepared_model)
        with torch.no_grad():
            traced_model = torch.jit.trace(converted_model, new_x)
            traced_model = torch.jit.freeze(traced_model)
        result_ref = traced_model(new_x)
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
        per_channel_observer = torch.ao.quantization.MovingAveragePerChannelMinMaxObserver
        custom_config = {
            'alpha' : 0.75,
            'act_observer' : torch.ao.quantization.MinMaxObserver(),
            'act_ic_observer' : per_channel_observer(ch_axis=-1),
            'wei_observer' : per_channel_observer(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
            'wei_ic_observer' : per_channel_observer(ch_axis=1),
        }
        for use_custom_config in [False, True]:
            kwargs = custom_config if use_custom_config else {}
            qconfig_mapping = ipex.quantization.get_smooth_quant_qconfig_mapping(**kwargs)
            prepared_model = ipex.quantization.prepare(
                m, qconfig_mapping, example_inputs=x, inplace=False)

            # Save observer info for comparison
            if use_custom_config:
                observer_info = {
                    **prepared_model._fqn_to_auto_quant_state_map[' '].tensor_id_to_observer,
                    **prepared_model._fqn_to_auto_quant_state_map[' '].weight_tensor_id_to_observer
                }
                observer_info_dict = {}
                for key, obs in observer_info.items():
                    observer_info_dict[key] = {
                        'smooth_quant_enabled' : obs.smooth_quant_enabled,
                        'alpha' : obs.alpha,
                        'ic_obs' : type(obs.ic_obs),
                        'act_obs' : type(obs.act_obs)
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

                prepared_model_2 = ipex.quantization.prepare(m, qconfig_mapping, example_inputs=x, inplace=False)
                prepared_model_2.load_qconf_summary(qconf_summary=qconf_filename)

                # Save observer info for comparison
                if use_custom_config:
                    observer_info_2 = {
                        **prepared_model_2._fqn_to_auto_quant_state_map[' '].tensor_id_to_observer,
                        **prepared_model_2._fqn_to_auto_quant_state_map[' '].weight_tensor_id_to_observer
                    }
                    observer_info_dict_2 = {}
                    for key, obs in observer_info_2.items():
                        observer_info_dict_2[key] = {
                            'smooth_quant_enabled' : obs.smooth_quant_enabled,
                            'alpha' : obs.alpha,
                            'ic_obs' : type(obs.ic_obs),
                            'act_obs' : type(obs.act_obs)
                        }

                q_model_2 = ipex.quantization.convert(prepared_model_2)

                with torch.no_grad():
                    q_model_2 = torch.jit.trace(q_model_2, x)
                    q_model_2 = torch.jit.freeze(q_model_2)
                out_2 = q_model_2(x)

                assert torch.allclose(out_ref, out_2)

            # Check observers
            if use_custom_config:
                assert observer_info_dict == observer_info_dict_2, \
                    "Error: SmoothQuant observer info lost after saving/loading qconf JSON"

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
        assert hasattr(converted_model, 'linear')
        assert isinstance(converted_model.linear, nn.quantized.dynamic.Linear)

        # Static quant
        qconfig_mapping = ipex.quantization.default_static_qconfig_mapping
        with self.assertRaises(AssertionError):
            prepared_model = ipex.quantization.prepare(m, qconfig_mapping)

    def test_weight_only_quantization_path(self):
        class M(nn.Module):
            def __init__(self, use_bias):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(4, 4, use_bias)

            def forward(self, x):
                return self.linear(x)

        for use_bias in [True, False]:
            m = M(use_bias).eval()
            x = torch.rand(4, 4)
            qconfig = ipex.quantization.weight_only_quant_qconfig_mapping
            prepared_model = ipex.quantization.prepare(m, qconfig, example_inputs=x, inplace=False)
            woq_model = ipex.quantization.convert(prepared_model)
            woq_model(x)
            woq_linear_class = ipex.nn.modules.weight_only_quantization.IpexWoqLinear
            assert isinstance(woq_model.linear, woq_linear_class)

    def test_weight_only_quantization_autocast(self):
        class M(nn.Module):
            def __init__(self, use_bias):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(4, 4, use_bias)

            def forward(self, x):
                return self.linear(x)

        with torch.autocast(device_type='cpu', enabled=True, dtype= torch.bfloat16):
            for use_bias in [True, False]:
                m = M(use_bias).eval()
                x = torch.rand(4, 4)
                qconfig = ipex.quantization.weight_only_quant_qconfig_mapping
                prepared_model = ipex.quantization.prepare(m, qconfig, example_inputs=x, inplace=False)
                woq_model = ipex.quantization.convert(prepared_model)
                woq_model(x)
                woq_linear_class = ipex.nn.modules.weight_only_quantization.IpexWoqLinear
                assert isinstance(woq_model.linear, woq_linear_class)

if __name__ == '__main__':
    run_tests() 
