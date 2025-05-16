import sys
import os
import tempfile
import unittest

from intel_extension_for_pytorch.llm.utils import (
    load_low_precision_checkpoint,
    shard_low_precision_checkpoint,
)
import torch
import torch.nn as nn
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing import FileCheck
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.nn.utils._weight_prepack import (
    may_import_deepspeed_modules,
    _IPEXLinear,
    _IPEXLinearAllreduce,
    _IPEXLmHeadLinearAllreduce,
)
from intel_extension_for_pytorch.quantization import prepare, convert
from intel_extension_for_pytorch.quantization._quantize import (
    DynamicQuantizedLinearLayer,
    DynamicQuantizedLinearAllreduce,
    DynamicQuantizedLmHeadLinearAllreduce,
)
from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
    _enable_tpp,
    _disable_tpp,
)

from test_weight_prepack import module_found
import json

try:
    import transformers
    from transformers import AutoConfig
except ImportError:
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "transformers==4.48.0"]
    )
    import transformers
    from transformers import AutoConfig


class MyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # For deepspeed support, please do not change the name of the attribute.
        self.q_proj = nn.Linear(64, 128)
        self.out_proj = nn.Linear(128, 128)

    def forward(self, x):
        x = self.q_proj(x)
        z = self.out_proj(x)
        return z


class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MyAttention()

    def forward(self, x):
        z = self.attn(x)
        return z


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # For deepspeed support, please do not change the ModuleList structure of the class.
        self.linears = nn.ModuleList([MyBlock()])

    def forward(self, x):
        for l in self.linears:
            x = l(x)
        return x


# For deepspeed support, please do not change the name of the class.
class MyLmHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        # For deepspeed support, please do not change the ModuleList structure of the class.
        self.linears = nn.ModuleList([MyBlock()])

    def forward(self, x):
        for l in self.linears:
            x = l(x)
        return x


# The class DeepSpeedTestM is written for deepspeed to recognize the modules and to be functional.
# Please do not change it.
class DeepSpeedTestM(nn.Module):
    def __init__(self, module_type):
        super().__init__()
        self.linear = module_type()
        self.lm_head = nn.Linear(128, 100)

    def forward(self, x):
        x = self.linear(x)
        x = self.lm_head(x)
        return x


class GPTJAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4096, 4096, bias=False)
        self.out_proj = nn.Linear(4096, 4096, bias=False)

    def forward(self, x):
        x = self.q_proj(x)
        z = self.out_proj(x)
        return z


class GPTJMLP(nn.Module):
    def __init__(self, krnl="tpp"):
        super().__init__()
        self.krnl = krnl
        self.fc_in = nn.Linear(4096, 16384, bias=True)
        self.fc_out = nn.Linear(16384, 4096, bias=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        if self.krnl == "onednn":
            x = self.fc_in(x)
            x = nn.functional.gelu(x, approximate="tanh")
        else:
            x = torch.ops.torch_ipex.tpp_linear_gelu(
                x, self.fc_in.weight, self.fc_in.bias
            )
        x = self.fc_out(x)
        x = self.dropout(x)
        return x


class GPTJBlock(nn.Module):
    def __init__(self, krnl):
        super().__init__()
        self.ln = nn.LayerNorm(4096, eps=1e-05)
        self.attn = GPTJAttention()
        self.mlp = GPTJMLP(krnl)

    def forward(self, x):
        x = self.ln(x)
        y = self.attn(x)
        z = self.mlp(x)
        x = y + z + x
        return x


class GPTJModel(nn.Module):
    def __init__(self, krnl):
        super().__init__()
        self.linears = nn.ModuleList([GPTJBlock(krnl)])

    def forward(self, x):
        for l in self.linears:
            x = l(x)
        return x


class GPTJTestM(nn.Module):
    def __init__(self, krnl):
        super().__init__()
        self.linear = GPTJModel(krnl)

    def forward(self, x):
        z = self.linear(x)
        return z


class DeepspeedTester(JitTestCase):
    def _get_ds_model(self, m_linear):
        import deepspeed

        ds_world_size = int(os.getenv("WORLD_SIZE", "1"))
        assert (
            ds_world_size > 1
        ), "expect ds_world_size > 1, you could try launching the script with: \
            deepspeed --num_gpus 2 --bind_cores_to_rank tests/cpu/test_deepspeed.py"
        engine = deepspeed.init_inference(
            model=m_linear,
            mp_size=ds_world_size,
            dtype=torch.float32,
            replace_method="auto",
        )
        ds_model = engine.module
        return ds_model

    def test_ipex_optimize(self):
        deepspeed_modules = may_import_deepspeed_modules()
        if deepspeed_modules is not None:
            LinearAllreduce, LinearLayer, LmHeadLinearAllreduce = deepspeed_modules

            x = torch.randn(2, 3, 64)
            m_linear = DeepSpeedTestM(MyLmHeadModel).eval()
            y = m_linear(x)

            ds_model = self._get_ds_model(m_linear)
            self.assertTrue(module_found(ds_model, LinearLayer))
            self.assertTrue(module_found(ds_model, LinearAllreduce))
            self.assertTrue(module_found(ds_model, LmHeadLinearAllreduce))

            optimized = ipex.optimize(
                ds_model.eval(),
                inplace=True,
                conv_bn_folding=False,
                linear_bn_folding=False,
            )

            with torch.no_grad():
                y_optimized = optimized(x)
                self.assertEqual(y, y_optimized)

                jit_optimized = torch.jit.trace(optimized, x)
                jit_optimized = torch.jit.freeze(jit_optimized)
                self.assertTrue(module_found(optimized, _IPEXLinear))
                self.assertTrue(module_found(optimized, _IPEXLinearAllreduce))
                self.assertTrue(module_found(optimized, _IPEXLmHeadLinearAllreduce))

                jit_optimized(x)
                graph = jit_optimized.graph_for(x)
                jit_res = jit_optimized(x)
                self.assertEqual(y, jit_res)

    def _test_quantization(
        self,
        dynamic_qconfig,
        qmodules,
        lm_head_qmodules,
        graph_strings,
        atol=0.005,
        rtol=1.3e-6,
    ):
        deepspeed_modules = may_import_deepspeed_modules()
        if deepspeed_modules is not None:
            LinearAllreduce, LinearLayer, LmHeadLinearAllreduce = deepspeed_modules

            x = torch.randn(2, 3, 64)
            m_linear = DeepSpeedTestM(MyLmHeadModel).eval()
            y = m_linear(x)

            ds_model = self._get_ds_model(m_linear)
            y_ds = ds_model(x)
            self.assertEqual(y, y_ds, atol=atol, rtol=rtol)
            self.assertTrue(module_found(ds_model, LinearLayer))
            self.assertTrue(module_found(ds_model, LinearAllreduce))
            self.assertTrue(module_found(ds_model, LmHeadLinearAllreduce))

            prepared_model = prepare(
                ds_model,
                dynamic_qconfig,
                example_inputs=(x),
                inplace=True,
                bn_folding=False,
            )
            converted = convert(prepared_model, inplace=True)
            self.assertTrue(
                all(module_found(converted, qmodule) for qmodule in qmodules)
            )
            prepared_model_ref = prepare(
                m_linear,
                dynamic_qconfig,
                example_inputs=(x),
                inplace=True,
                bn_folding=False,
            )
            converted_ref = convert(prepared_model_ref, inplace=True)

            self.assertTrue(
                all(
                    module_found(converted, lm_head_qmodule)
                    for lm_head_qmodule in lm_head_qmodules
                )
            )

            with torch.no_grad():
                y_quantized = converted(x)
                y_ref = converted_ref(x)
                self.assertEqual(y_ref, y_quantized, atol=atol, rtol=rtol)

                converted = torch.jit.trace(converted, x)
                traced = torch.jit.freeze(converted)

                traced(x)  # profiling run
                graph = traced.graph_for(x)
                for graph_string in graph_strings:
                    FileCheck().check(graph_string).run(graph)

                y_traced = traced(x)
                self.assertEqual(y_ref, y_traced, atol=atol, rtol=rtol)

                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "ds_model.pt")

                    torch.jit.save(traced, path)
                    loaded = torch.jit.load(path)

                    loaded(x)  # profiling run
                    graph_loaded = loaded.graph_for(x)
                    for graph_string in graph_strings:
                        FileCheck().check(graph_string).run(graph_loaded)

                    y_loaded = loaded(x)
                    self.assertEqual(y_ref, y_loaded, atol=atol, rtol=rtol)

    def test_dynamic_quantization(self):
        self._test_quantization(
            ipex.quantization.default_dynamic_qconfig,
            [DynamicQuantizedLinearLayer, DynamicQuantizedLinearAllreduce],
            [DynamicQuantizedLmHeadLinearAllreduce],
            ["quantized::linear_dynamic", "deepspeed_comm::all_reduce"],
            atol=0.02,
            rtol=1.3e-6,
        )

    def test_weight_only_quantization(self):
        self._test_quantization(
            ipex.quantization.get_weight_only_quant_qconfig_mapping(),
            [
                ipex.nn.modules.weight_only_quantization.WeightOnlyQuantizedLinear,
                ipex.nn.modules.weight_only_quantization.IpexWoqLinearAllreduce,
            ],
            [ipex.nn.modules.weight_only_quantization.IpexWoqLmHeadLinearAllreduce],
            ["torch_ipex::ipex_woq_linear", "deepspeed_comm::all_reduce"],
        )

    def test_simplify_allreduce_for_gptj(self):
        deepspeed_modules = may_import_deepspeed_modules()
        if deepspeed_modules is not None:
            ds_pattern = "deepspeed_comm::all_reduce"
            x = torch.rand(4, 32, 4096)
            for krnl in ["onednn", "tpp"]:
                m = GPTJTestM(krnl).eval()
                ds_model = self._get_ds_model(m)
                if krnl == "tpp":
                    _enable_tpp()
                optimized = ipex.optimize(
                    ds_model.eval(),
                    inplace=True,
                    auto_kernel_selection=True if krnl == "onednn" else False,
                    conv_bn_folding=False,
                    linear_bn_folding=False,
                )
                with torch.no_grad():
                    y = optimized(x)
                    jit_optimized = torch.jit.trace(
                        optimized, x, strict=False, check_trace=False
                    )
                    jit_optimized = torch.jit.freeze(jit_optimized)
                    graph = jit_optimized.graph_for(x)
                    self.assertGraphContainsExactly(graph, ds_pattern, 2)
                    jit_optimized(x)
                    graph = jit_optimized.graph_for(x)
                    self.assertGraphContainsExactly(graph, ds_pattern, 1)
                    jit_res = jit_optimized(x)
                    self.assertEqual(y, jit_res)
                _disable_tpp()

    def test_llama_with_llm_optimize(self):
        curpath = os.path.abspath(os.path.dirname(__file__))
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/llama", return_dict=False
        )
        model = transformers.models.llama.modeling_llama.LlamaForCausalLM(config).eval()
        model = self._get_ds_model(model)
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping()
        model = ipex.llm.optimize(
            model.eval(),
            dtype=torch.bfloat16,
            quantization_config=qconfig,
            inplace=True,
            deployment_mode=True,
        )
        if not hasattr(model, "trace_graph"):
            AssertionError(False)
        _IPEXAttentionCPU = (
            ipex.transformers.models.cpu.modules.attentions._IPEXAttentionCPU
        )
        _IPEXDecoderLayerCPU = (
            ipex.transformers.models.cpu.modules.decoder._IPEXDecoderLayerCPU
        )
        assert model.model.layers[0].self_attn.__class__ is _IPEXAttentionCPU
        assert model.model.layers[0].__class__ is _IPEXDecoderLayerCPU
        # Ensure model can run without errors
        input_ids = torch.ones(8).to(torch.long)
        attention_mask = torch.ones(len(input_ids))
        position_ids = torch.arange(len(input_ids))
        past_key_values = tuple(
            [
                (
                    torch.zeros(1, 1, 0, 1, dtype=torch.long).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros(1, 4, dtype=torch.long),
                )
                for i in range(1)
            ]
        )
        example_inputs = (
            input_ids.unsqueeze(0),
            attention_mask.unsqueeze(0),
            past_key_values,
            position_ids.unsqueeze(0),
        )
        with torch.no_grad():
            model(*example_inputs)

    def test_shard_awq_low_precision_checkpoint(self):
        curpath = os.path.abspath(os.path.dirname(__file__))
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/llama", return_dict=False
        )
        ds_world_size = int(os.getenv("WORLD_SIZE", "1"))
        if ds_world_size != 2:
            print("Warning: Expect ds_world_size to be 2.")
            return
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        model = transformers.models.llama.modeling_llama.LlamaForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as work_dir:
            # Generate dummy checkpoint
            checkpoint_file_name = work_dir + "/checkpoint.pt"
            state_dict = model.state_dict()
            linear_keys = []
            for k, v in state_dict.items():
                if any(
                    k.endswith(suffix)
                    for suffix in ["proj.weight", "fc_in.weight", "fc_out.weight"]
                ):
                    linear_keys.append(k[:-7])
            group_size = 128
            comp_ratio = 8
            for k in linear_keys:
                N = state_dict[k + ".weight"].shape[0]
                K = state_dict[k + ".weight"].shape[1]
                del state_dict[k + ".weight"]
                n_groups = K // group_size
                stored_weight_shape = (K, N // comp_ratio)
                stored_scales_shape = (n_groups, N)
                stored_zeros_shape = (n_groups, N // comp_ratio)
                state_dict[k + ".qweight"] = torch.randint(
                    -(2**31), 2**31 - 1, stored_weight_shape, dtype=torch.int32
                )
                state_dict[k + ".scales"] = torch.randn(
                    stored_scales_shape, dtype=torch.half
                )
                state_dict[k + ".qzeros"] = torch.randint(
                    -(2**31), 2**31 - 1, stored_zeros_shape, dtype=torch.int32
                )

            torch.save(state_dict, checkpoint_file_name)
            quantization_config = {
                "quant_method": "awq",
                "group_size": group_size,
                "desc_act": False,
            }
            config_dict = {"quantization_config": quantization_config}
            config_file_name = work_dir + "/config.json"
            with open(config_file_name, "w", encoding="utf-8") as file:
                json.dump(config_dict, file, ensure_ascii=False, indent=4)
            low_precision_checkpoint, quant_config = load_low_precision_checkpoint(
                work_dir
            )
            quantization_method = quant_config["quant_method"]
            group_size = quant_config["group_size"]
            quantization_method = "awq"
            sharded_low_precision_checkpoint = shard_low_precision_checkpoint(
                low_precision_checkpoint,
                config.to_dict(),
                local_rank,
                ds_world_size,
                quantization_method,
                group_size,
                desc_act=False,
                bits=4,
            )
            sharded_low_precision_checkpoint = (
                sharded_low_precision_checkpoint,
                quant_config,
            )
        shard_dict = {
            "self_attn.q_proj.qweight": [4096, 256],
            "self_attn.q_proj.scales": [32, 2048],
            "self_attn.q_proj.zeros": [32, 256],
            "self_attn.k_proj.qweight": [4096, 256],
            "self_attn.k_proj.scales": [32, 2048],
            "self_attn.k_proj.zeros": [32, 256],
            "self_attn.v_proj.qweight": [4096, 256],
            "self_attn.v_proj.scales": [32, 2048],
            "self_attn.v_proj.zeros": [32, 256],
            "self_attn.o_proj.qweight": [2048, 512],
            "self_attn.o_proj.scales": [16, 4096],
            "self_attn.o_proj.zeros": [16, 512],
            "mlp.up_proj.qweight": [4096, 688],
            "mlp.up_proj.scales": [32, 5504],
            "mlp.up_proj.zeros": [32, 688],
            "mlp.gate_proj.qweight": [4096, 688],
            "mlp.gate_proj.scales": [32, 5504],
            "mlp.gate_proj.zeros": [32, 688],
            "mlp.down_proj.qweight": [5504, 512],
            "mlp.down_proj.scales": [43, 4096],
            "mlp.down_proj.zeros": [43, 512],
            "lm_head.weight": [32000, 2048],
        }
        assert quantization_method == "awq"
        low_precision_checkpoint_dict = sharded_low_precision_checkpoint[0].copy()
        for key in low_precision_checkpoint_dict.keys():
            for layer in shard_dict:
                if layer not in key:
                    continue
                if "bias" in key:
                    continue
                assert (
                    low_precision_checkpoint_dict[key].shape[0] == shard_dict[layer][0]
                    and low_precision_checkpoint_dict[key].shape[1]
                    == shard_dict[layer][1]
                ), "shape after shard does not match"

    def test_shard_gptq_low_precision_checkpoint(self):
        curpath = os.path.abspath(os.path.dirname(__file__))
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/llama", return_dict=False
        )
        ds_world_size = int(os.getenv("WORLD_SIZE", "1"))
        if ds_world_size != 2:
            print("Warning: Expect ds_world_size to be 2.")
            return
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        model = transformers.models.llama.modeling_llama.LlamaForCausalLM(config).eval()
        for desc_act in {True, False}:
            with tempfile.TemporaryDirectory() as work_dir:
                # Generate dummy checkpoint
                checkpoint_file_name = work_dir + "/checkpoint.pt"
                state_dict = model.state_dict()
                linear_keys = []
                for k, v in state_dict.items():
                    if any(
                        k.endswith(suffix)
                        for suffix in ["proj.weight", "fc_in.weight", "fc_out.weight"]
                    ):
                        linear_keys.append(k[:-7])
                group_size = 32
                comp_ratio = 8
                for k in linear_keys:
                    N = state_dict[k + ".weight"].shape[0]
                    K = state_dict[k + ".weight"].shape[1]
                    del state_dict[k + ".weight"]
                    n_groups = K // group_size
                    stored_weight_shape = (K, N // comp_ratio)
                    stored_scales_shape = (n_groups, N)
                    stored_zeros_shape = (n_groups, N // comp_ratio)
                    stored_g_idx_shape = (K * 8,)
                    state_dict[k + ".qweight"] = torch.randint(
                        -(2**31), 2**31 - 1, stored_weight_shape, dtype=torch.int32
                    )
                    state_dict[k + ".scales"] = torch.randn(
                        stored_scales_shape, dtype=torch.half
                    )
                    state_dict[k + ".qzeros"] = torch.randint(
                        -(2**31), 2**31 - 1, stored_zeros_shape, dtype=torch.int32
                    )
                    state_dict[k + ".g_idx"] = torch.randint(
                        -(2**31), 2**31 - 1, stored_g_idx_shape, dtype=torch.int32
                    )

                torch.save(state_dict, checkpoint_file_name)
                quantization_config = {
                    "quant_method": "gptq",
                    "group_size": group_size,
                    "desc_act": desc_act,
                }
                config_dict = {"quantization_config": quantization_config}
                config_file_name = work_dir + "/config.json"
                with open(config_file_name, "w", encoding="utf-8") as file:
                    json.dump(config_dict, file, ensure_ascii=False, indent=4)
                low_precision_checkpoint, quant_config = load_low_precision_checkpoint(
                    work_dir
                )
                quantization_method = quant_config["quant_method"]
                group_size = quant_config["group_size"]
                sharded_low_precision_checkpoint = shard_low_precision_checkpoint(
                    low_precision_checkpoint,
                    config.to_dict(),
                    local_rank,
                    ds_world_size,
                    quantization_method,
                    group_size,
                    desc_act,
                    bits=4,
                )
                sharded_low_precision_checkpoint = (
                    sharded_low_precision_checkpoint,
                    quant_config,
                )
            desc_true_shard_dict = {
                "self_attn.q_proj.qweight": [4096, 256],
                "self_attn.q_proj.scales": [128, 2048],
                "self_attn.q_proj.zeros": [128, 256],
                "self_attn.q_proj.g_idx": [32768],
                "self_attn.k_proj.qweight": [4096, 256],
                "self_attn.k_proj.scales": [128, 2048],
                "self_attn.k_proj.zeros": [128, 256],
                "self_attn.k_proj.g_idx": [32768],
                "self_attn.v_proj.qweight": [4096, 256],
                "self_attn.v_proj.scales": [128, 2048],
                "self_attn.v_proj.zeros": [128, 256],
                "self_attn.v_proj.g_idx": [32768],
                "self_attn.o_proj.qweight": [2048, 512],
                "self_attn.o_proj.scales": [128, 4096],
                "self_attn.o_proj.zeros": [128, 512],
                "self_attn.o_proj.g_idx": [16384],
                "mlp.up_proj.qweight_0": [4096, 704],
                "mlp.up_proj.qweight_1": [4096, 672],
                "mlp.up_proj.scales": [128, 5504],
                "mlp.up_proj.zeros_0": [128, 704],
                "mlp.up_proj.zeros_1": [128, 672],
                "mlp.up_proj.g_idx": [32768],
                "mlp.gate_proj.qweight_0": [4096, 704],
                "mlp.gate_proj.qweight_1": [4096, 672],
                "mlp.gate_proj.scales": [128, 5504],
                "mlp.gate_proj.zeros_0": [128, 704],
                "mlp.gate_proj.zeros_1": [128, 672],
                "mlp.gate_proj.g_idx": [32768],
                "mlp.down_proj.qweight": [5504, 512],
                "mlp.down_proj.scales": [344, 4096],
                "mlp.down_proj.zeros": [344, 512],
                "mlp.down_proj.g_idx": [44032],
                "lm_head.weight": [32000, 2048],
            }
            desc_false_shard_dict = {
                "self_attn.q_proj.qweight": [4096, 256],
                "self_attn.q_proj.scales": [128, 2048],
                "self_attn.q_proj.zeros": [128, 256],
                "self_attn.k_proj.qweight": [4096, 256],
                "self_attn.k_proj.scales": [128, 2048],
                "self_attn.k_proj.zeros": [128, 256],
                "self_attn.v_proj.qweight": [4096, 256],
                "self_attn.v_proj.scales": [128, 2048],
                "self_attn.v_proj.zeros": [128, 256],
                "self_attn.o_proj.qweight": [2048, 512],
                "self_attn.o_proj.scales": [64, 4096],
                "self_attn.o_proj.zeros": [64, 512],
                "mlp.up_proj.qweight_0": [4096, 704],
                "mlp.up_proj.qweight_1": [4096, 672],
                "mlp.up_proj.scales": [128, 5504],
                "mlp.up_proj.zeros_1": [128, 704],
                "mlp.up_proj.zeros_0": [128, 672],
                "mlp.gate_proj.qweight_0": [4096, 704],
                "mlp.gate_proj.qweight_1": [4096, 672],
                "mlp.gate_proj.scales": [128, 5504],
                "mlp.gate_proj.zeros_0": [128, 704],
                "mlp.gate_proj.zeros_1": [128, 672],
                "mlp.down_proj.qweight": [5504, 512],
                "mlp.down_proj.scales": [172, 4096],
                "mlp.down_proj.zeros": [172, 512],
                "lm_head.weight": [32000, 2048],
            }
            if desc_act is True:
                shard_dict = desc_true_shard_dict
            else:
                shard_dict = desc_false_shard_dict
            assert quantization_method == "gptq"
            low_precision_checkpoint_dict = sharded_low_precision_checkpoint[0].copy()
            for key in low_precision_checkpoint_dict.keys():
                for layer in shard_dict:
                    if layer not in key:
                        continue
                    if "bias" in key:
                        continue
                    if "g_idx" in key:
                        assert (
                            low_precision_checkpoint_dict[key].shape[0]
                            == shard_dict[layer][0]
                        ), "shape after shard does not match"
                    else:
                        if ("up_proj" in layer or "gate_proj" in layer) and (
                            "qweight" in key or "zeros" in key
                        ):
                            layer = layer + "_" + str(local_rank)
                        assert (
                            low_precision_checkpoint_dict[key].shape[0]
                            == shard_dict[layer][0]
                            and low_precision_checkpoint_dict[key].shape[1]
                            == shard_dict[layer][1]
                        ), "shape after shard does not match"


if __name__ == "__main__":
    deepspeed_modules = may_import_deepspeed_modules()
    if deepspeed_modules is not None:
        # when launching with deepspeed, the cmd will be python -u tests/cpu/test_deepspeed.py --local_rank=xx
        # Need to handle the --local_rank before unittest.main()
        if len(sys.argv) > 1:
            local_rank = sys.argv.pop()

        test = unittest.main()
