import sys
import os
import tempfile
import unittest

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

try:
    import transformers
    from transformers import AutoConfig
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "transformers==4.38.1"]
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
            LinearAllreduce, LinearLayer = deepspeed_modules[:2]
            # TODO: remove check_lm_head logic once deepspeed LmHeadLinearAllreduce change has been upstream-ed.
            check_lm_head = False
            if len(deepspeed_modules) == 3:
                check_lm_head = True
                LmHeadLinearAllreduce = deepspeed_modules[2]

            x = torch.randn(2, 3, 64)
            m_linear = DeepSpeedTestM(MyLmHeadModel).eval()
            y = m_linear(x)

            ds_model = self._get_ds_model(m_linear)
            self.assertTrue(module_found(ds_model, LinearLayer))
            self.assertTrue(module_found(ds_model, LinearAllreduce))

            # TODO: [Notes: LmHeadLinearAllreduce replacement test]
            # On the public master of deepspeed,
            # LmHeadLinearAllreduce replacement will only happen if checkpoint has been loaded:
            # github.com/microsoft/DeepSpeed/blob/16c265c0ce103147d027d9cae32dd7680766af21/deepspeed/module_inject/replace_module.py
            # #L352
            # Need to figure out a way to use checkpoint in the UT to test LmHeadLinearAllreduce replacement.
            # Disable the check for now.
            if False:  # if check_lm_head:
                self.assertTrue(module_found(ds_model, LmHeadLinearAllreduce))

            optimized = ipex.optimize(ds_model.eval(), inplace=True)

            with torch.no_grad():
                y_optimized = optimized(x)
                self.assertEqual(y, y_optimized)

                jit_optimized = torch.jit.trace(optimized, x)
                jit_optimized = torch.jit.freeze(jit_optimized)
                self.assertTrue(module_found(optimized, _IPEXLinear))
                self.assertTrue(module_found(optimized, _IPEXLinearAllreduce))

                # TODO: Check [Notes: LmHeadLinearAllreduce replacement test]
                if False:  # if check_lm_head:
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
            LinearAllreduce, LinearLayer = deepspeed_modules[:2]
            # TODO: remove check_lm_head logic once deepspeed LmHeadLinearAllreduce change has been upstream-ed.
            check_lm_head = False
            if len(deepspeed_modules) == 3:
                check_lm_head = True
                LmHeadLinearAllreduce = deepspeed_modules[2]

            x = torch.randn(2, 3, 64)
            m_linear = DeepSpeedTestM(MyLmHeadModel).eval()
            y = m_linear(x)

            ds_model = self._get_ds_model(m_linear)
            self.assertTrue(module_found(ds_model, LinearLayer))
            self.assertTrue(module_found(ds_model, LinearAllreduce))
            # TODO: Check [Notes: LmHeadLinearAllreduce replacement test]
            if False:  # if check_lm_head:
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

            # TODO: Check [Notes: LmHeadLinearAllreduce replacement test]
            if False:  # if check_lm_head
                self.assertTrue(
                    all(
                        module_found(converted, lm_head_qmodule)
                        for lm_head_qmodule in lm_head_qmodules
                    )
                )

            with torch.no_grad():
                y_quantized = converted(x)
                self.assertEqual(y, y_quantized, atol=atol, rtol=rtol)

                converted = torch.jit.trace(converted, x)
                traced = torch.jit.freeze(converted)

                traced(x)  # profiling run
                graph = traced.graph_for(x)
                for graph_string in graph_strings:
                    FileCheck().check(graph_string).run(graph)

                y_traced = traced(x)
                self.assertEqual(y, y_traced, atol=atol, rtol=rtol)

                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "ds_model.pt")

                    torch.jit.save(traced, path)
                    loaded = torch.jit.load(path)

                    loaded(x)  # profiling run
                    graph_loaded = loaded.graph_for(x)
                    for graph_string in graph_strings:
                        FileCheck().check(graph_string).run(graph_loaded)

                    y_loaded = loaded(x)
                    self.assertEqual(y, y_loaded, atol=atol, rtol=rtol)

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


if __name__ == "__main__":
    deepspeed_modules = may_import_deepspeed_modules()
    if deepspeed_modules is not None:
        # when launching with deepspeed, the cmd will be python -u tests/cpu/test_deepspeed.py --local_rank=xx
        # Need to handle the --local_rank before unittest.main()
        if len(sys.argv) > 1:
            local_rank = sys.argv.pop()

        test = unittest.main()
