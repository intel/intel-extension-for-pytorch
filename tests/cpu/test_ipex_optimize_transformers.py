import unittest
import torch
import intel_extension_for_pytorch as ipex
import sys
import subprocess
import os
import copy
import re
import tempfile
from intel_extension_for_pytorch.quantization import prepare, convert
from collections import namedtuple
import itertools

try:
    import transformers
    from transformers import AutoConfig
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "transformers==4.38.1"]
    )
    import transformers
    from transformers import AutoConfig
from intel_extension_for_pytorch.cpu._auto_kernel_selection import _disable_tpp

from common_utils import TestCase

torch.manual_seed(128)

curpath = os.path.abspath(os.path.dirname(__file__))


def _get_gptj_example_inputs():
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
    return (
        input_ids.unsqueeze(0),
        attention_mask.unsqueeze(0),
        past_key_values,
        position_ids.unsqueeze(0),
    )


model_info = namedtuple(
    "model_info",
    "name, model_class, has_position_ids, attention_class, decoder_class",
)
supported_models = [
    model_info(
        "gptj",
        transformers.models.gptj.modeling_gptj.GPTJForCausalLM,
        True,
        lambda m: m.transformer.h[0].attn.__class__,
        lambda m: m.transformer.h[0].__class__,
    ),
    model_info(
        "llama",
        transformers.models.llama.modeling_llama.LlamaForCausalLM,
        True,
        lambda m: m.model.layers[0].self_attn.__class__,
        lambda m: m.model.layers[0].__class__,
    ),
]


class OptimizeTransformersTester(TestCase):
    def model_replacement_check(
        self, m, dtype, deployment_mode, torchcompile=False, return_dict=False
    ):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/{m.name}",
            return_dict=return_dict,
            trust_remote_code=True,
        )
        model = m.model_class(config).eval()
        if m.name == "falcon":
            with torch.no_grad():
                ipex.nn.utils._model_convert.replace_customized_linear_with_linear(
                    model.eval()
                )
        elif m.name == "chatglm":
            state_dict = model.state_dict()
            for weight in [
                "transformer.encoder.layers.0.input_layernorm.weight",
                "transformer.encoder.layers.0.post_attention_layernorm.weight",
                "transformer.encoder.final_layernorm.weight",
            ]:
                state_dict[weight] = torch.rand(state_dict[weight].shape)
            model.load_state_dict(state_dict)
        elif m.name == "baichuan":
            state_dict = model.state_dict()
            for weight in [
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.norm.weight",
            ]:
                state_dict[weight] = torch.rand(state_dict[weight].shape)
            model.load_state_dict(state_dict)
        model.eval()
        ref_m = copy.deepcopy(model)
        ipex_m = copy.deepcopy(model)
        ipex_m = ipex.llm.optimize(
            ipex_m, dtype=dtype, deployment_mode=deployment_mode, inplace=True
        )
        if torchcompile:
            torch._dynamo.reset()
            ipex._set_compiler_backend("inductor")
            ipex_m = torch.compile(ipex_m, backend="ipex")

        assert (
            m.attention_class(ipex_m)
            is ipex.transformers.models.cpu.modules.attentions._IPEXAttentionCPU
        )
        assert (
            m.decoder_class(ipex_m)
            is ipex.transformers.models.cpu.modules.decoder._IPEXDecoderLayerCPU
            if m.decoder_class is not None
            else True
        )

        input_ids = torch.ones(10).to(torch.long)
        attention_mask = torch.ones(len(input_ids))
        position_ids = torch.arange(len(input_ids))
        decoder_input_ids = torch.ones(1).to(torch.long)
        input_dict = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "use_cache": True,
        }
        if m.has_position_ids:
            input_dict["position_ids"] = position_ids.unsqueeze(0)
        if re.search("t5", model.config.architectures[0], re.IGNORECASE):
            input_dict["decoder_input_ids"] = decoder_input_ids.unsqueeze(0)
        with torch.no_grad():
            key_hf = ref_m(**input_dict)
        with torch.no_grad(), torch.cpu.amp.autocast(
            enabled=True if dtype is torch.bfloat16 else False
        ):
            key_ipex = ipex_m(**input_dict)
        error_message = f"model={m.name}, deployment_mode={deployment_mode}, torchcompile={torchcompile}, return_dict={return_dict}"
        if return_dict:
            assert isinstance(key_ipex, dict)
            self.assertEqual(
                key_hf["logits"], key_ipex["logits"], prec=0.1, message=error_message
            )
        else:
            assert isinstance(key_ipex, tuple)
            self.assertEqual(key_hf[0], key_ipex[0], prec=0.1, message=error_message)

    def test_model_replacement(self):
        dtypes = [torch.bfloat16]
        enable_torchcompile = [False, True]
        deployment_mode = [True, False]
        return_dict = [False, True]
        for m, torchcompile, dtype, jit, return_dict in itertools.product(
            supported_models, enable_torchcompile, dtypes, deployment_mode, return_dict
        ):
            if torchcompile and deployment_mode:
                continue
            self.model_replacement_check(m, dtype, jit, torchcompile, return_dict)
        _disable_tpp()

    def _model_replacement_check_woq(self, model):
        qconfig_mapping = ipex.quantization.get_weight_only_quant_qconfig_mapping()
        orig_model = copy.deepcopy(model)
        orig_woq_model = prepare(orig_model, qconfig_mapping, inplace=True)
        orig_woq_model = convert(orig_woq_model, inplace=True)

        model = ipex.llm.optimize(
            model,
            dtype=torch.float,
            quantization_config=qconfig_mapping,
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
        WeightOnlyQuantizedLinear = ipex.nn.modules.WeightOnlyQuantizedLinear
        if re.search("GPTJ", model.config.architectures[0]):
            assert model.transformer.h[0].attn.__class__ is _IPEXAttentionCPU
            assert model.transformer.h[0].__class__ is _IPEXDecoderLayerCPU
            assert all(
                mod.__class__ is WeightOnlyQuantizedLinear
                for mod in [
                    model.transformer.h[0].attn.concat_qkv.concat_linear,
                    model.transformer.h[0].attn.out_proj,
                    model.transformer.h[0].linear_add_add.linear,
                    model.transformer.h[0].linear_gelu.linear,
                ]
            )
        elif re.search("llama", model.config.architectures[0], re.IGNORECASE):
            assert model.model.layers[0].self_attn.__class__ is _IPEXAttentionCPU
            assert model.model.layers[0].__class__ is _IPEXDecoderLayerCPU
            assert all(
                mod.__class__ is WeightOnlyQuantizedLinear
                for mod in [
                    model.model.layers[0].self_attn.concat_qkv.concat_linear,
                    model.model.layers[0].mha_linear_add.linear,
                    model.model.layers[0].mlp_linear_add.linear,
                    model.model.layers[0].linear_silu_mul.linear_s,
                    model.model.layers[0].linear_silu_mul.linear_m,
                ]
            )
        # Ensure model can run without errors
        with torch.no_grad():
            example_inputs = _get_gptj_example_inputs()
            y = model(*example_inputs)
            y_ref = orig_woq_model(
                input_ids=example_inputs[0],
                attention_mask=example_inputs[1],
                position_ids=example_inputs[3],
                use_cache=True,
            )
            self.assertEqual(y[0], y_ref[0], prec=1e-4)

    def test_weight_only_quant_flow_for_gptj(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/gptj", return_dict=False
        )
        m = transformers.models.gptj.modeling_gptj.GPTJForCausalLM(config).eval()
        self._model_replacement_check_woq(m)

    def test_weight_only_quant_flow_for_llama(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/llama", return_dict=False
        )
        m = transformers.models.llama.modeling_llama.LlamaForCausalLM(config).eval()
        self._model_replacement_check_woq(m)

    def test_static_quant_flow(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/gptj", return_dict=False
        )
        m = transformers.models.gptj.modeling_gptj.GPTJForCausalLM(config).eval()
        quant_m = copy.deepcopy(m)
        qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping()
        example_inputs = _get_gptj_example_inputs()
        quant_m = ipex.llm.optimize(
            quant_m, dtype=torch.float, quantization_config=qconfig, inplace=True
        )
        from intel_extension_for_pytorch.quantization import prepare

        prepared_model = prepare(
            quant_m.eval(), qconfig, example_inputs=example_inputs, inplace=True
        )

        with torch.no_grad():
            prepared_model(*example_inputs)
        with tempfile.NamedTemporaryFile() as fp:
            prepared_model.save_qconf_summary(qconf_summary=fp.name)

            for dtype in [torch.float, torch.bfloat16]:
                ipex_m = copy.deepcopy(m)
                if dtype is torch.bfloat16:
                    ipex_m = ipex_m.to(torch.bfloat16)
                qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping()
                ipex_m = ipex.llm.optimize(
                    ipex_m,
                    dtype=dtype,
                    quantization_config=qconfig,
                    qconfig_summary_file=fp.name,
                    inplace=True,
                )
                if not hasattr(ipex_m, "trace_graph"):
                    AssertionError(False)

    def test_weight_only_quant_gptq_legacy(self):
        # Test the legacy format
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/gptj", return_dict=False
        )
        m = transformers.models.gptj.modeling_gptj.GPTJForCausalLM(config).eval()
        ipex_m = copy.deepcopy(m)
        with tempfile.TemporaryDirectory() as work_dir:
            # Generate dummy checkpoint
            checkpoint_file_name = work_dir + "/checkpoint.pt"
            state_dict = ipex_m.state_dict()
            linear_keys = []
            for k, v in state_dict.items():
                if any(
                    k.endswith(suffix)
                    for suffix in ["proj.weight", "fc_in.weight", "fc_out.weight"]
                ):
                    linear_keys.append(k[:-7])
            for k in linear_keys:
                N = state_dict[k + ".weight"].shape[0]
                K = state_dict[k + ".weight"].shape[1]
                del state_dict[k + ".weight"]
                state_dict[k + ".packed_weight"] = torch.randint(
                    -(2**31), 2**31 - 1, (N, K // 8), dtype=torch.int32
                )
                state_dict[k + ".scale"] = torch.ones((N, 1), dtype=torch.half) * 0.5
                state_dict[k + ".packed_zp"] = torch.ones((N, 1), dtype=torch.int32) * 4

            torch.save(state_dict, checkpoint_file_name)
            state_dict = torch.load(checkpoint_file_name)

            # test loading checkpoint and quant info
            lowp_mode = ipex.quantization.WoqLowpMode.INT8
            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                lowp_mode=lowp_mode
            )
            config_dict = (
                ipex.utils.weight_only_quantization._legacy_lowp_checkpoint_config()
            )
            ipex_m = ipex.llm.optimize(
                ipex_m,
                dtype=torch.float,
                quantization_config=qconfig,
                low_precision_checkpoint=(state_dict, config_dict),
                deployment_mode=True,
                inplace=True,
            )
            assert hasattr(ipex_m, "trace_graph")

            # Ensure model can run without errors
            with torch.no_grad():
                example_inputs = _get_gptj_example_inputs()
                # the optimized model is ipex_m.trace_graph
                ipex_m.trace_graph(*example_inputs)

    def test_weight_only_quant_gptq(self):
        # Test the HuggingFace Optimum format
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/gptj", return_dict=False
        )
        m = transformers.models.gptj.modeling_gptj.GPTJForCausalLM(config).eval()
        ipex_m = copy.deepcopy(m)
        with tempfile.TemporaryDirectory() as work_dir:
            # Generate dummy checkpoint
            checkpoint_file_name = work_dir + "/checkpoint.pt"
            state_dict = ipex_m.state_dict()
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
                stored_weight_shape = (K // comp_ratio, N)
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
                g_idx = torch.arange(n_groups).repeat(group_size)
                g_idx[:] = g_idx[torch.randperm(K)]
                state_dict[k + ".g_idx"] = g_idx

            torch.save(state_dict, checkpoint_file_name)
            state_dict = torch.load(checkpoint_file_name)

            # test loading checkpoint and quant info
            lowp_mode = ipex.quantization.WoqLowpMode.INT8
            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                lowp_mode=lowp_mode
            )
            ipex_m = ipex.llm.optimize(
                ipex_m,
                dtype=torch.float,
                quantization_config=qconfig,
                low_precision_checkpoint=state_dict,
                deployment_mode=True,
                inplace=True,
            )
            assert hasattr(ipex_m, "trace_graph")

            # Ensure model can run without errors
            with torch.no_grad():
                example_inputs = _get_gptj_example_inputs()
                # the optimized model is ipex_m.trace_graph
                ipex_m.trace_graph(*example_inputs)

    def test_generate_functions(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/gptj", return_dict=False
        )
        m = transformers.models.gptj.modeling_gptj.GPTJForCausalLM(config).eval()
        ref_m = copy.deepcopy(m)
        ipex_m = ipex.llm.optimize(
            m, dtype=torch.bfloat16, deployment_mode=True, inplace=True
        )
        input_ids = torch.ones(8).unsqueeze(0).to(torch.long)
        # beam_search, beam=4
        generate_kwargs_beam = dict(
            do_sample=False,
            temperature=0.9,
            num_beams=4,
            max_new_tokens=2,
            min_new_tokens=2,
        )
        # greedy_search
        generate_kwargs_greedy = dict(
            do_sample=False, temperature=0.9, max_new_tokens=2, min_new_tokens=2
        )
        # sample, use a temperature of 0.01 to constrain text generation diversity in UT.
        generate_kwargs_sample = dict(
            do_sample=True, temperature=0.01, max_new_tokens=2, min_new_tokens=2
        )
        # beam_sample, use a temperature of 0.01 to constrain text generation diversity in UT.
        generate_kwargs_sample = dict(
            do_sample=True,
            temperature=0.01,
            num_beams=4,
            max_new_tokens=2,
            min_new_tokens=2,
        )
        for generate_kwargs in [
            generate_kwargs_beam,
            generate_kwargs_greedy,
            generate_kwargs_sample,
            generate_kwargs_sample,
        ]:
            with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast():
                ipex_res = ipex_m.generate(input_ids, **generate_kwargs)
                ref_res = ref_m.generate(input_ids, **generate_kwargs)
                self.assertEqual(ipex_res, ref_res)


if __name__ == "__main__":
    test = unittest.main()
