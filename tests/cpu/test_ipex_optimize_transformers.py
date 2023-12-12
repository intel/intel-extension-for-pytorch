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

try:
    import transformers
    from transformers import AutoConfig
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "transformers==4.31.0"]
    )
    import transformers
    from transformers import AutoConfig

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
        position_ids.unsqueeze(0),
        past_key_values,
    )


class OptimizeTransformersTester(TestCase):
    def model_replacement_check(self, model, has_position_id, torchcompile=False):
        for dtype in [torch.bfloat16]:
            for deployment_mode in [True, False]:
                if torchcompile and deployment_mode:
                    continue
                ref_m = copy.deepcopy(model)
                ipex_m = copy.deepcopy(model)

                ipex_m = ipex.optimize_transformers(
                    ipex_m, dtype=dtype, deployment_mode=deployment_mode, inplace=True
                )

                if torchcompile:
                    torch._dynamo.reset()
                    ipex._set_compiler_backend("inductor")
                    ipex_m = torch.compile(ipex_m, backend="ipex")

                input_ids = torch.ones(10).to(torch.long)
                attention_mask = torch.ones(len(input_ids))
                with torch.no_grad(), torch.cpu.amp.autocast(
                    enabled=True if dtype is torch.bfloat16 else False
                ):
                    if has_position_id:
                        position_ids = torch.arange(len(input_ids))
                        key_hf = ref_m(
                            input_ids=input_ids.unsqueeze(0),
                            attention_mask=attention_mask.unsqueeze(0),
                            position_ids=position_ids.unsqueeze(0),
                            use_cache=True,
                        )
                        key_ipex = ipex_m(
                            input_ids=input_ids.unsqueeze(0),
                            attention_mask=attention_mask.unsqueeze(0),
                            position_ids=position_ids.unsqueeze(0),
                            use_cache=True,
                        )
                    else:
                        key_hf = ref_m(
                            input_ids=input_ids.unsqueeze(0),
                            attention_mask=attention_mask.unsqueeze(0),
                            use_cache=True,
                        )
                        key_ipex = ipex_m(
                            input_ids=input_ids.unsqueeze(0),
                            attention_mask=attention_mask.unsqueeze(0),
                            use_cache=True,
                        )
                    self.assertEqual(key_hf[0], key_ipex[0], prec=0.1)

                    if re.search("GPTJ", model.config.architectures[0]) or re.search(
                        "codegen", model.config.architectures[0]
                    ):
                        assert (
                            ipex_m.transformer.h[0].attn.__class__
                            is ipex.transformers.models.cpu.modules.attentions._IPEXAttentionCPU
                        )
                        assert (
                            ipex_m.transformer.h[0].__class__
                            is ipex.transformers.models.cpu.modules.decoder._IPEXDecoderLayerCPU
                        )
                    elif re.search(
                        "llama", model.config.architectures[0], re.IGNORECASE
                    ):
                        assert (
                            ipex_m.model.layers[0].self_attn.__class__
                            is ipex.transformers.models.cpu.modules.attentions._IPEXAttentionCPU
                        )
                        assert (
                            ipex_m.model.layers[0].__class__
                            is ipex.transformers.models.cpu.modules.decoder._IPEXDecoderLayerCPU
                        )
                    elif re.search(
                        "gptneox", model.config.architectures[0], re.IGNORECASE
                    ):
                        assert (
                            ipex_m.gpt_neox.layers[0].attention.__class__
                            is ipex.transformers.models.cpu.modules.attentions._IPEXAttentionCPU
                        )
                    elif re.search("opt", model.config.architectures[0], re.IGNORECASE):
                        assert (
                            ipex_m.model.decoder.layers[0].self_attn.__class__
                            is ipex.transformers.models.cpu.modules.attentions._IPEXAttentionCPU
                        )
                        assert (
                            ipex_m.model.decoder.layers[0].__class__
                            is ipex.transformers.models.cpu.modules.decoder._IPEXDecoderLayerCPU
                        )
                    elif re.search(
                        "falcon", model.config.architectures[0], re.IGNORECASE
                    ) or re.search(
                        "bloom", model.config.architectures[0], re.IGNORECASE
                    ):
                        assert (
                            type(ipex_m.transformer.h[0].self_attention)
                            is ipex.transformers.models.cpu.modules.attentions._IPEXAttentionCPU
                        )
                        assert (
                            type(ipex_m.transformer.h[0])
                            is ipex.transformers.models.cpu.modules.decoder._IPEXDecoderLayerCPU
                        )
                    elif re.search(
                        "baichuan", model.config.architectures[0], re.IGNORECASE
                    ):
                        assert (
                            type(ipex_m.model.layers[0].self_attn)
                            is ipex.transformers.models.cpu.modules.attentions._IPEXAttentionCPU
                        )
                        assert (
                            type(ipex_m.model.layers[0])
                            is ipex.transformers.models.cpu.modules.decoder._IPEXDecoderLayerCPU
                        )
                    elif re.search(
                        "chatglm", model.config.architectures[0], re.IGNORECASE
                    ):
                        assert (
                            type(ipex_m.transformer.encoder.layers[0].self_attention)
                            is ipex.transformers.models.cpu.modules.attentions._IPEXAttentionCPU
                        )
                        assert (
                            type(ipex_m.transformer.encoder.layers[0])
                            is ipex.transformers.models.cpu.modules.decoder._IPEXDecoderLayerCPU
                        )

    def test_model_replacement_gptj(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/gptj", return_dict=False
        )
        m = transformers.models.gptj.modeling_gptj.GPTJForCausalLM(config).eval()
        self.model_replacement_check(m, True)

    def test_model_replacement_gptj_torchcompile(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/gptj", return_dict=False
        )
        m = transformers.models.gptj.modeling_gptj.GPTJForCausalLM(config).eval()
        self.model_replacement_check(m, True, torchcompile=True)

    def test_model_replacement_llama(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/llama", return_dict=False
        )
        m = transformers.models.llama.modeling_llama.LlamaForCausalLM(config).eval()
        self.model_replacement_check(m, True)

    def test_model_replacement_llama_torchcompile(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/llama", return_dict=False
        )
        m = transformers.models.llama.modeling_llama.LlamaForCausalLM(config).eval()
        self.model_replacement_check(m, True, torchcompile=True)

    def test_model_replacement_gptneox(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/gptneox", return_dict=False
        )
        m = transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM(
            config
        ).eval()
        self.model_replacement_check(m, True)

    def test_model_replacement_gptneox_torchcompile(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/gptneox", return_dict=False
        )
        m = transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM(
            config
        ).eval()
        self.model_replacement_check(m, True, torchcompile=True)

    def test_model_replacement_opt(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/opt", return_dict=False
        )

        m = transformers.models.opt.modeling_opt.OPTForCausalLM(config).eval()
        self.model_replacement_check(m, False)

    def test_model_replacement_opt_torchcompile(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/opt", return_dict=False
        )

        m = transformers.models.opt.modeling_opt.OPTForCausalLM(config).eval()
        self.model_replacement_check(m, False, torchcompile=True)

    def test_model_replacement_falcon(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/falcon", return_dict=False
        )

        m = transformers.models.falcon.modeling_falcon.FalconForCausalLM(config).eval()
        with torch.no_grad():
            ipex.nn.utils._model_convert.replace_customized_linear_with_linear(m.eval())
        self.model_replacement_check(m, False)

    def test_model_replacement_falcon_torchcompile(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/falcon", return_dict=False
        )

        m = transformers.models.falcon.modeling_falcon.FalconForCausalLM(config).eval()
        with torch.no_grad():
            ipex.nn.utils._model_convert.replace_customized_linear_with_linear(m.eval())
        self.model_replacement_check(m, False, torchcompile=True)

    def test_model_replacement_bloom(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/bloom", return_dict=False
        )

        m = transformers.models.bloom.modeling_bloom.BloomForCausalLM(config).eval()
        self.model_replacement_check(m, False)

    def test_model_replacement_bloom_torchcompile(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/bloom", return_dict=False
        )
        m = transformers.models.bloom.modeling_bloom.BloomForCausalLM(config).eval()
        # TODO: fix accuracy issue
        # self.model_replacement_check(m, False, torchcompile=True)

    def test_model_replacement_codegen(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/codegen", return_dict=False
        )
        m = transformers.models.codegen.modeling_codegen.CodeGenForCausalLM(
            config
        ).eval()
        self.model_replacement_check(m, True)

    def test_model_replacement_codegen_torchcompile(self):
        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/codegen", return_dict=False
        )
        m = transformers.models.codegen.modeling_codegen.CodeGenForCausalLM(
            config
        ).eval()
        self.model_replacement_check(m, True, torchcompile=True)

    def test_model_replacement_baichuan(self):
        from hf_configs.baichuan.modeling_baichuan import BaichuanForCausalLM

        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/baichuan", return_dict=False, trust_remote_code=True
        )
        m = BaichuanForCausalLM(config).eval()
        self.model_replacement_check(m, False)

    def test_model_replacement_baichuan_torchcompile(self):
        from hf_configs.baichuan.modeling_baichuan import BaichuanForCausalLM

        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/baichuan", return_dict=False, trust_remote_code=True
        )
        m = BaichuanForCausalLM(config).eval()
        self.model_replacement_check(m, False, torchcompile=True)

    def test_model_replacement_chatglm(self):
        from hf_configs.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration

        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/chatglm", return_dict=False, trust_remote_code=True
        )
        m = ChatGLMForConditionalGeneration(config).eval()
        self.model_replacement_check(m, False)

    def test_model_replacement_chatglm_torchcompile(self):
        from hf_configs.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration

        config = AutoConfig.from_pretrained(
            f"{curpath}/hf_configs/chatglm", return_dict=False, trust_remote_code=True
        )
        m = ChatGLMForConditionalGeneration(config).eval()
        self.model_replacement_check(m, False, torchcompile=True)

    def _model_replacement_check_woq(self, model):
        qconfig_mapping = ipex.quantization.get_weight_only_quant_qconfig_mapping()
        orig_model = copy.deepcopy(model)
        orig_woq_model = prepare(orig_model, qconfig_mapping, inplace=True)
        orig_woq_model = convert(orig_woq_model, inplace=True)

        model = ipex.optimize_transformers(
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
        IpexWoqLinear = ipex.nn.modules.IpexWoqLinear
        if re.search("GPTJ", model.config.architectures[0]):
            assert model.transformer.h[0].attn.__class__ is _IPEXAttentionCPU
            assert model.transformer.h[0].__class__ is _IPEXDecoderLayerCPU
            assert all(
                mod.__class__ is IpexWoqLinear
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
                mod.__class__ is IpexWoqLinear
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
                position_ids=example_inputs[2],
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
        quant_m = ipex.optimize_transformers(
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
                ipex_m = ipex.optimize_transformers(
                    ipex_m,
                    dtype=dtype,
                    quantization_config=qconfig,
                    qconfig_summary_file=fp.name,
                    inplace=True,
                )
                if not hasattr(ipex_m, "trace_graph"):
                    AssertionError(False)

    def test_weight_only_quant_gptq(self):
        # import json
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
            ipex_m = ipex.optimize_transformers(
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


if __name__ == "__main__":
    test = unittest.main()
