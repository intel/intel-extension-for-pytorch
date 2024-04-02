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
import unittest
import numpy
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
)


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
                self.dense2 = nn.Linear(4, 4)

            def forward(self, x):
                return self.dense2(self.relu(self.dense(x)))

        m = Mod().eval()
        x = torch.rand(1, 4)
        calib_dataset = [torch.rand(1, 4) for _ in range(5)]
        per_channel_observer = (
            torch.ao.quantization.MovingAveragePerChannelMinMaxObserver
        )
        custom_config = {
            "alpha": 0.75,
            "act_observer": torch.ao.quantization.MinMaxObserver,
            "act_ic_observer": per_channel_observer.with_args(ch_axis=-1),
            "wei_observer": per_channel_observer.with_args(
                dtype=torch.qint8, qscheme=torch.per_channel_symmetric
            ),
            "wei_ic_observer": per_channel_observer.with_args(ch_axis=1),
            "share_weight_observers": False,
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
                observer_info_dict["share_weight_observers"] = (
                    prepared_model._fqn_to_auto_quant_state_map[" "]
                    .idx_to_seen_q_op_infos[0]
                    .qconfig.share_weight_observers
                )
                sub_observer_ids = {
                    "act_ic_obs": [],
                    "act_obs": [],
                    "wei_oc_obs": [],
                    "wei_ic_obs": [],
                }
                for key, obs in observer_info.items():
                    observer_info_dict[key] = {
                        "smooth_quant_enabled": obs.smooth_quant_enabled,
                        "alpha": obs.alpha,
                        "ic_obs": type(obs.ic_obs),
                        "act_obs": type(obs.act_obs),
                    }
                    if isinstance(
                        obs,
                        ipex.quantization._smooth_quant.SmoothQuantActivationObserver,
                    ):
                        sub_observer_ids["act_ic_obs"].append(id(obs.ic_obs))
                        sub_observer_ids["act_obs"].append(id(obs.act_obs))
                    else:
                        sub_observer_ids["wei_oc_obs"].append(id(obs.oc_obs))
                        sub_observer_ids["wei_ic_obs"].append(id(obs.ic_obs))
                for _, id_list in sub_observer_ids.items():
                    assert all([id_list[0] != id for id in id_list[1:]])

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
                    observer_info_dict_2["share_weight_observers"] = (
                        prepared_model_2._fqn_to_auto_quant_state_map[" "]
                        .idx_to_seen_q_op_infos[0]
                        .qconfig.share_weight_observers
                    )
                    sub_observer_ids = {
                        "act_ic_obs": [],
                        "act_obs": [],
                        "wei_oc_obs": [],
                        "wei_ic_obs": [],
                    }
                    for key, obs in observer_info_2.items():
                        observer_info_dict_2[key] = {
                            "smooth_quant_enabled": obs.smooth_quant_enabled,
                            "alpha": obs.alpha,
                            "ic_obs": type(obs.ic_obs),
                            "act_obs": type(obs.act_obs),
                        }
                        if isinstance(
                            obs,
                            ipex.quantization._smooth_quant.SmoothQuantActivationObserver,
                        ):
                            sub_observer_ids["act_ic_obs"].append(id(obs.ic_obs))
                            sub_observer_ids["act_obs"].append(id(obs.act_obs))
                        else:
                            sub_observer_ids["wei_oc_obs"].append(id(obs.oc_obs))
                            sub_observer_ids["wei_ic_obs"].append(id(obs.ic_obs))
                    for _, id_list in sub_observer_ids.items():
                        assert all([id_list[0] != id for id in id_list[1:]])

                q_model_2 = ipex.quantization.convert(prepared_model_2)

                with torch.no_grad():
                    q_model_2 = torch.jit.trace(q_model_2, x)
                    q_model_2 = torch.jit.freeze(q_model_2)
                out_2 = q_model_2(x)

                assert torch.allclose(out_ref, out_2)

                # Scales and zero points should be updated after rerunning calibration
                scale_zp_0 = prepared_model_2._fqn_to_auto_quant_state_map[
                    " "
                ].tensor_id_to_scale_zp
                scale_zp_0 = copy.deepcopy(scale_zp_0)
                for data in calib_dataset:
                    prepared_model_2(data + 1)
                prepared_model_2.save_qconf_summary(qconf_summary=qconf_filename)
                scale_zp_1 = prepared_model_2._fqn_to_auto_quant_state_map[
                    " "
                ].tensor_id_to_scale_zp
                assert scale_zp_0 != scale_zp_1

            # Check observers
            if use_custom_config:
                assert (
                    observer_info_dict == observer_info_dict_2
                ), "Error: SmoothQuant observer info lost after saving/loading qconf JSON"

    def test_smooth_quant_cancel_by_qconf_summary(self):
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
        qconfig_mapping = ipex.quantization.get_smooth_quant_qconfig_mapping()
        prepared_model = ipex.quantization.prepare(
            m, qconfig_mapping, example_inputs=x, inplace=False
        )
        for data in calib_dataset:
            prepared_model(data)

        with tempfile.NamedTemporaryFile() as fp:
            qconf_filename = fp.name
            prepared_model.save_qconf_summary(qconf_summary=qconf_filename)
            import json

            with open(qconf_filename, "r") as qconf_file:
                parsed = json.load(qconf_file)
                parsed[" "]["q_op_infos"]["0"]["input_tensor_infos"][0][
                    "force_dtype"
                ] = "torch.float32"

            with open(qconf_filename, "w") as qconf_file:
                json.dump(parsed, qconf_file, indent=4)

            prepared_model_2 = ipex.quantization.prepare(
                m, qconfig_mapping, example_inputs=x, inplace=False
            )
            prepared_model_2.load_qconf_summary(qconf_summary=qconf_filename)
            converted_model = ipex.quantization.convert(prepared_model_2)
            with torch.no_grad():
                jit_model = torch.jit.trace(converted_model, x)
                jit_model = torch.jit.freeze(jit_model)
                for _ in range(2):
                    jit_model(x)
                graph = jit_model.graph_for(x)
                for n in graph.nodes():
                    assert n.kind() != "aten::mul"

    def test_smooth_quant_share_weight_observers(self):
        class Mod(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(4, 4)
                self.k_proj = nn.Linear(4, 4)
                self.v_proj = nn.Linear(4, 4)
                self.relu = nn.ReLU()

            def forward(self, x):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                return self.relu(torch.concat([q, k, v], axis=1))

        m = Mod().eval()
        x = torch.rand(1, 4)
        calib_dataset = [torch.rand(1, 4) for _ in range(5)]
        for share_weight_observers in [True, False]:
            qconfig_mapping = ipex.quantization.get_smooth_quant_qconfig_mapping(
                share_weight_observers=share_weight_observers
            )
            prepared_model = ipex.quantization.prepare(
                m, qconfig_mapping, example_inputs=x, inplace=True
            )
            for data in calib_dataset:
                prepared_model(data)
            q_model = ipex.quantization.convert(prepared_model)
            with torch.no_grad():
                q_model = torch.jit.trace(q_model, x)
                q_model = torch.jit.freeze(q_model)
                graph = q_model.graph_for(x)
                num_mul = [n.kind() for n in graph.nodes()].count("aten::mul")
                assert num_mul == 1 if share_weight_observers else 3
                q_model(x)

    def test_smooth_quant_autotune(self):
        class DemoModel(torch.nn.Module):
            def __init__(self):
                super(DemoModel, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.fc2(out)
                return out

        class DemoCalibDataloader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                yield torch.randn([1, 3])

        m = DemoModel().eval()
        calib_dataloader = DemoCalibDataloader()
        inputs = torch.rand(1, 3)

        def calib_func(model):
            model(inputs)

        smoothquant_args_global = {
            "alpha": numpy.arange(0.0, 1.0, 0.1).tolist(),
        }
        smoothquant_args_layer = {
            "alpha": "auto",
            "auto_alpha_args": {
                "init_alpha": 0.8,
                "alpha_min": 0.8,
                "alpha_max": 0.99,
                "alpha_step": 0.01,
                "shared_criterion": "mean",
                "enable_blockwise_loss": False,
            },
        }

        for folding in [True, False]:
            for smoothquant_args in [
                smoothquant_args_global,
                smoothquant_args_layer,
            ]:
                model = copy.deepcopy(m)
                sq_args = copy.deepcopy(smoothquant_args)
                smoothquant_args["folding"] = folding
                tuned_model = ipex.quantization.autotune(
                    model,
                    calib_dataloader,
                    eval_func=lambda x: 0.1,
                    smoothquant_args=sq_args,
                    sampling_sizes=[100],
                    accuracy_criterion={"relative": 0.01},
                    tuning_time=0,
                )
                converted_model = ipex.quantization.convert(tuned_model)
                with torch.no_grad():
                    traced_model = torch.jit.trace(converted_model, inputs)
                    traced_model = torch.jit.freeze(traced_model)
                    y = traced_model(inputs)

                model = copy.deepcopy(m)
                sq_args = copy.deepcopy(smoothquant_args)
                smoothquant_args["folding"] = folding
                tuned_model = ipex.quantization.autotune(
                    model,
                    calib_dataloader,
                    calib_func=calib_func,
                    smoothquant_args=sq_args,
                    sampling_sizes=[100],
                    accuracy_criterion={"relative": 0.01},
                    tuning_time=0,
                )
                converted_model = ipex.quantization.convert(tuned_model)
                with torch.no_grad():
                    traced_model = torch.jit.trace(converted_model, inputs)
                    traced_model = torch.jit.freeze(traced_model)
                    y = traced_model(inputs)

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
                woq_linear_class = (
                    ipex.nn.modules.weight_only_quantization.WeightOnlyQuantizedLinear
                )
                assert isinstance(woq_model.linear, woq_linear_class)
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

            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=w_dtype
            )
            prepared_model = prepare(m, qconfig, example_inputs=data, inplace=False)

            with torch.no_grad():
                weight = m.linear.weight
                weight_int8, w_scales, w_zero_points = quantize_per_channel(
                    weight, w_dtype
                )
                weight_fp32 = dequantize_per_channel(
                    weight_int8, w_scales, w_zero_points.int(), w_dtype, weight.shape
                )
                weight_bf16 = weight_fp32.bfloat16()
                weight_fp16 = weight_fp32.half()
                data_bf16 = data.bfloat16()
                data_fp16 = data_bf16.half()
                bias_fp32 = m.linear.bias
                # if M >= 32, compute in bf16
                # if M < 32, compute in fp32 or fp16. Depends on fp16 support.
                if feature[0] >= 32:
                    output1 = torch.matmul(
                        data_bf16.float(), weight_bf16.float().T
                    ).bfloat16()
                    if has_bias:
                        output1 = output1 + bias_fp32.bfloat16()
                else:
                    output1_fp32 = torch.matmul(
                        data_bf16.float(), weight_bf16.float().T
                    )
                    if has_bias:
                        output1_fp32 = output1_fp32 + bias_fp32
                    output1_fp16 = torch.matmul(
                        data_fp16.float(), weight_fp16.float().T
                    ).half()
                    if has_bias:
                        output1_fp16 = output1_fp16 + bias_fp32.half()
                with torch.autocast(
                    device_type="cpu", enabled=True, dtype=torch.bfloat16
                ):
                    woq_model = convert(prepared_model)
                    woq_linear_class = (
                        ipex.nn.modules.weight_only_quantization.WeightOnlyQuantizedLinear
                    )
                    assert isinstance(woq_model.linear, woq_linear_class)

                    woq_model = torch.jit.trace(woq_model, data)
                    woq_model = torch.jit.freeze(woq_model)
                    output2 = woq_model(data)
                    output2 = output2.bfloat16()
                if feature[0] < 32:
                    try:
                        torch.testing.assert_close(
                            output1_fp32.bfloat16(), output2, atol=0.01, rtol=0.1
                        )
                    except Exception as e:
                        torch.testing.assert_close(
                            output1_fp16.bfloat16(), output2, atol=0.01, rtol=0.1
                        )
                else:
                    torch.testing.assert_close(output1, output2)

        shape_list = [
            [3, 31, 31],
            [4, 64, 64],
            [9, 128, 128],
            [196, 63, 255],
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
            [9, 4095, 4095],
            [196, 4095, 16383],
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
                woq_linear_class = (
                    ipex.nn.modules.weight_only_quantization.WeightOnlyQuantizedLinear
                )
                assert isinstance(woq_model.linear, woq_linear_class)
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
                weight, WoqWeightDtype.NF4
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
                woq_linear_class = (
                    ipex.nn.modules.weight_only_quantization.WeightOnlyQuantizedLinear
                )
                assert isinstance(woq_model.linear, woq_linear_class)
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
        ]
        use_bias_list = [True, False]
        cases = itertools.product(shape_list, use_bias_list)
        for shape, use_bias in cases:
            test(shape, use_bias)

    def test_weight_only_quantization_gelu_fused_op(self):
        class Mod(nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = nn.Linear(64, 64, bias=bias)
                self.gelu = nn.GELU()

            def forward(self, x):
                return self.gelu(self.linear(x))

        bias_list = [False, True]
        bf16_list = [False, True]
        cases = itertools.product(bias_list, bf16_list)
        for bias, bf16 in cases:
            with torch.cpu.amp.autocast(
                enabled=bf16, dtype=torch.bfloat16 if bf16 else None
            ):
                model = Mod(bias).eval()
                data = torch.rand(4, 64)
                qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                    lowp_mode=2
                )
                prepared_model = prepare(
                    model, qconfig, example_inputs=data, inplace=False
                )
                with torch.no_grad():
                    woq_model = convert(prepared_model)
                    output1 = woq_model(data)
                    output2 = torch.ops.torch_ipex.woq_linear_gelu(
                        data, woq_model.linear._op_context.get_data_handle()
                    )
                    torch.testing.assert_close(
                        output1, output2.to(output1.dtype), atol=1e-2, rtol=1e-4
                    )

    def test_weight_only_quantization_new_gelu_fused_op(self):
        class Mod(nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = nn.Linear(64, 64, bias=bias)
                self.gelu = nn.GELU(approximate="tanh")

            def forward(self, x):
                return self.gelu(self.linear(x))

        bias_list = [False, True]
        bf16_list = [False, True]
        cases = itertools.product(bias_list, bf16_list)
        for bias, bf16 in cases:
            with torch.cpu.amp.autocast(
                enabled=bf16, dtype=torch.bfloat16 if bf16 else None
            ):
                model = Mod(bias).eval()
                data = torch.rand(4, 64)
                qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                    lowp_mode=2
                )
                prepared_model = prepare(
                    model, qconfig, example_inputs=data, inplace=False
                )
                with torch.no_grad():
                    woq_model = convert(prepared_model)
                    output1 = woq_model(data)
                    output2 = torch.ops.torch_ipex.woq_linear_new_gelu(
                        data, woq_model.linear._op_context.get_data_handle()
                    )
                    torch.testing.assert_close(
                        output1, output2.to(output1.dtype), atol=1e-2, rtol=1e-4
                    )

    def test_weight_only_quantization_add_fused_op(self):
        class Mod(nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = nn.Linear(64, 64, bias=bias)

            def forward(self, x, others):
                y = self.linear(x)
                for o in others:
                    y = torch.add(y, o)
                return y

        bias_list = [False, True]
        bf16_list = [False, True]
        others_len_list = [1, 2]
        cases = itertools.product(bias_list, bf16_list, others_len_list)
        for bias, bf16, others_len in cases:
            with torch.cpu.amp.autocast(
                enabled=bf16, dtype=torch.bfloat16 if bf16 else None
            ):
                model = Mod(bias).eval()
                data = torch.rand(4, 64)
                others = [torch.rand(4, 64)] * others_len
                fused_op = (
                    torch.ops.torch_ipex.woq_linear_add
                    if others_len == 1
                    else torch.ops.torch_ipex.woq_linear_add_add
                )
                qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                    lowp_mode=2
                )
                prepared_model = prepare(
                    model, qconfig, example_inputs=data, inplace=False
                )
                with torch.no_grad():
                    woq_model = convert(prepared_model)
                    output1 = woq_model(data, others)
                    output2 = fused_op(
                        data, woq_model.linear._op_context.get_data_handle(), others
                    )
                    torch.testing.assert_close(
                        output1, output2.to(output1.dtype), atol=1.5e-2, rtol=1e-3
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

        # When lowp_mode=BF16, only case of batch size >= 32 uses BF16.
        data = torch.rand(32, 64)
        m = M()

        lowp_mode_list = [WoqLowpMode.NONE, WoqLowpMode.FP16, WoqLowpMode.BF16]
        act_dtype_list = [torch.bfloat16, torch.half]
        compute_dtype_list = [None, torch.half, torch.bfloat16]
        cases = itertools.product(lowp_mode_list, act_dtype_list)
        # lowp_mode does not affect weight observer for int8
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping()
        weight = copy.deepcopy(m.linear.weight)
        w_dtype = qconfig.global_qconfig.weight_dtype
        weight_int8, w_scales, w_zps = quantize_per_channel(weight, w_dtype)
        weight_fp32 = dequantize_per_channel(weight_int8, w_scales, w_zps, w_dtype)
        bias_fp32 = copy.deepcopy(m.linear.bias)
        for lowp_mode, act_dtype in cases:
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

    def test_weight_only_quantization_act_quant_mode(self):
        M, N, K = 4, 64, 128
        groupsize = 64

        class Mod(nn.Module):
            def __init__(self, has_bias):
                super(Mod, self).__init__()
                self.linear = torch.nn.Linear(K, N, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(has_bias, act_quant_mode):
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

        has_bias_list = [False, True]
        quant_mode_list = [0, 1, 2, 3]
        cases = itertools.product(has_bias_list, quant_mode_list)
        for has_bias, quant_mode in cases:
            test(has_bias, quant_mode)

    def test_weight_only_quantization_group_size(self):
        class Mod(nn.Module):
            def __init__(self, ic, oc, has_bias):
                super(Mod, self).__init__()
                self.linear = torch.nn.Linear(ic, oc, has_bias)

            def forward(self, x):
                return self.linear(x)

        def test(shape, has_bias, act_quant_mode, group_size):
            M, N, K = shape
            dtype = torch.bfloat16
            model = Mod(K, N, has_bias)
            m = model.eval()
            m2 = copy.deepcopy(m)
            data = torch.rand(M, K) * 0.5
            if group_size == -1 and act_quant_mode != 0:
                # these cases are covered by another test case for act_quant_mode
                return
            qconfig_mapping = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=WoqWeightDtype.INT4,
                lowp_mode=WoqLowpMode.INT8,
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
                if group_size == -1:
                    qw, w_scales, w_zero_points = quantize_per_channel(
                        w, WoqWeightDtype.INT4, None, None
                    )
                    fake_quant_w = dequantize_per_channel(
                        qw, w_scales, w_zero_points.int(), WoqWeightDtype.INT4, w.shape
                    )
                else:
                    qw, w_scales, w_zero_points = quantize_per_block(
                        w, WoqWeightDtype.INT4, group_size, None, None
                    )
                    fake_quant_w = dequantize_per_block(
                        qw,
                        w_scales,
                        w_zero_points,
                        WoqWeightDtype.INT4,
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

        MNK_list = [(4, 64, 128), (4, 32, 127), (9, 31, 256)]
        has_bias_list = [False, True]
        quant_mode_list = [0, 1, 2, 3]
        group_size_list = [-1, 32, 64, 128]
        cases = itertools.product(
            MNK_list, has_bias_list, quant_mode_list, group_size_list
        )
        for shape, has_bias, act_quant_mode, group_size in cases:
            test(shape, has_bias, act_quant_mode, group_size)

    def test_compute_with_g_idx(self):
        class Mod(nn.Module):
            def __init__(self, ic, oc, has_bias):
                super(Mod, self).__init__()
                self.linear = torch.nn.Linear(ic, oc, has_bias)

            def forward(self, x):
                return self.linear(x)

        shape_list = [[1, 32, 32], [16, 64, 64], [32, 128, 128]]
        group_size_list = [4, 16]
        cases = itertools.product(shape_list, group_size_list)
        for shape, group_size in cases:
            bs, ic, oc = shape
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
            x = torch.randn((bs, ic), dtype=torch.float)
            for has_bias in [True, False]:
                # woq path
                m = Mod(ic=ic, oc=oc, has_bias=has_bias)
                b = m.linear.bias.detach() if has_bias else None
                qconfig_mapping = (
                    ipex.quantization.get_weight_only_quant_qconfig_mapping(
                        weight_dtype=WoqWeightDtype.INT4,
                        lowp_mode=ipex.quantization.WoqLowpMode.INT8,
                        act_quant_mode=ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
                        group_size=group_size,
                    )
                )
                woq_m = copy.deepcopy(m)
                woq_m.linear.qconfig = qconfig_mapping.global_qconfig
                woq_m.linear = ipex.nn.modules.WeightOnlyQuantizedLinear.from_float_and_int4_weight(
                    woq_m.linear,
                    packed_weight,
                    scales,
                    packed_zeros,
                    b,
                    group_size=group_size,
                    g_idx=g_idx,
                )
                y = woq_m(x)

                # ref path
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
                scales_expanded = scales.repeat(1, group_size)
                zeros_expanded = zeros.repeat(1, group_size)
                dqw = (int4_weight.to(torch.float) - zeros_expanded) * scales_expanded
                y_ref = torch.nn.functional.linear(fqx, dqw, bias=b)
                y_ref_2 = torch.nn.functional.linear(x, dqw, bias=b)

                # check results
                try:
                    torch.testing.assert_close(y, y_ref, atol=1e-4, rtol=1e-5)
                except Exception:
                    # In IPEX CI, UT will run with different ISA
                    # This check is for the ref kernel, where x is not quantized
                    torch.testing.assert_close(y, y_ref_2, atol=1e-4, rtol=1e-5)

    def test_unpack_with_g_idx(self):
        class Mod(nn.Module):
            def __init__(self, ic, oc, has_bias):
                super(Mod, self).__init__()
                self.linear = torch.nn.Linear(ic, oc, has_bias)

            def forward(self, x):
                return self.linear(x)

        shape_list = [[64, 64], [256, 256]]
        group_size_list = [4, 16]
        cases = itertools.product(shape_list, group_size_list)
        for shape, group_size in cases:
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
                        lowp_mode=ipex.quantization.WoqLowpMode.INT8,
                        act_quant_mode=ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
                        group_size=group_size,
                    )
                )
                scales_expanded = scales.repeat(1, group_size)
                zeros_expanded = zeros.repeat(1, group_size)
                # path with g_idx
                woq_m = copy.deepcopy(m)
                woq_m.linear.qconfig = qconfig_mapping.global_qconfig
                woq_m.linear = ipex.nn.modules.WeightOnlyQuantizedLinear.from_float_and_int4_weight(
                    woq_m.linear,
                    packed_weight,
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
                woq_m_2.linear = ipex.nn.modules.WeightOnlyQuantizedLinear.from_float_and_int4_weight(
                    woq_m_2.linear,
                    packed_weight,
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


class QuantizedOpsTester(TestCase):
    def test_matmul_i8i8i32(self):
        x = torch.randn(4, 8)
        w = torch.randn(4, 8)
        x_min, x_max = x.aminmax()
        x_scale = torch.max(x_max, x_min.neg()) / 127
        qx = torch.round(x / x_scale).to(torch.int8)
        w_min, w_max = w.aminmax(dim=1)
        w_scale = torch.max(w_max, w_min.neg()) / 127
        qw = torch.round(w / w_scale.unsqueeze(-1)).to(torch.int8)
        for use_bf16 in [False, True]:
            dtype = torch.bfloat16 if use_bf16 else torch.float32
            with torch.cpu.amp.autocast(enabled=use_bf16, dtype=dtype):
                qy = torch.ops.torch_ipex.matmul_i8i8i32(qx, qw)
                qy_ref = torch.nn.functional.linear(qx.to(dtype), qw.to(dtype))
                self.assertEqual(qy.to(dtype), qy_ref)


if __name__ == "__main__":
    test = unittest.main()
    run_tests()
