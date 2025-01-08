import torch
from torch.testing._internal.common_utils import (
    TestCase,
)
import importlib
import intel_extension_for_pytorch as ipex
import os
import pytest
import transformers
from transformers import AutoConfig

torch.manual_seed(128)
curpath = os.path.abspath(os.path.dirname(__file__))
max_token_id = 1024
output_check_len = 8
long_sequence_len = 512

common_params = {
    "use_static_cache": [True, False],
    "num_beams": [1, 4],
    "input_tokens_length": [32, 1024],
    "max_new_tokens": [long_sequence_len],
}

need_recover_models_list = {
    transformers.models.llama.modeling_llama.LlamaForCausalLM: "LlamaModel",
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM: "Qwen2Model",
}


def cache_module(model_list):
    module_cache = {}
    for model_type, recovery_str in model_list.items():
        module_cache[model_type] = {
            "_update_causal_mask": None,
            "_prepare_4d_causal_attention_mask_with_cache_position": None,
        }
        base_module = model_type.__base__.__module__
        module_spec = importlib.import_module(base_module)
        model_spec = getattr(module_spec, recovery_str)
        if hasattr(model_spec, "_update_causal_mask"):
            module_cache[model_type][
                "_update_causal_mask"
            ] = model_spec._update_causal_mask
        if hasattr(
            module_spec, "_prepare_4d_causal_attention_mask_with_cache_position"
        ):
            module_cache[model_type][
                "_prepare_4d_causal_attention_mask_with_cache_position"
            ] = module_spec._prepare_4d_causal_attention_mask_with_cache_position

    return module_cache


def reload_module(cached_module, model):
    model_type = type(model)
    crt_module = model_type.__base__.__module__
    module_spec = importlib.import_module(crt_module)
    recovery_str = need_recover_models_list.get(model_type)
    if recovery_str is None:
        return
    model_spec = getattr(module_spec, recovery_str)
    if cached_module[model_type]["_update_causal_mask"] is not None:
        model_spec._update_causal_mask = cached_module[model_type][
            "_update_causal_mask"
        ]
    if (
        cached_module[model_type][
            "_prepare_4d_causal_attention_mask_with_cache_position"
        ]
        is not None
    ):
        module_spec._prepare_4d_causal_attention_mask_with_cache_position = (
            cached_module[model_type][
                "_prepare_4d_causal_attention_mask_with_cache_position"
            ]
        )


class TestIpexLLMOptimizeBase(TestCase):
    model_class = None
    model_config = None

    @classmethod
    def setUpClass(cls):
        super(TestIpexLLMOptimizeBase, cls).setUpClass()
        if cls.model_class is None or cls.model_config is None:
            raise ValueError("Sub class must define model_class and model_config")
        if cls.model_class in need_recover_models_list.keys():
            cls.cached_module = cache_module(
                {cls.model_class: need_recover_models_list.get(cls.model_class)}
            )

    @pytest.mark.skipif(
        (not torch.xpu.has_xetla()) or (not torch.xpu.has_2d_block_array()),
        reason="ipex build without xetla or is atsm",
    )
    def run_generate_test(
        self,
        use_static_cache,
        input_tokens_length,
        max_new_tokens,
        num_beams,
    ):
        config = AutoConfig.from_pretrained(
            self.model_config["config_path"], return_dict=False
        )
        dtype = torch.float16
        device = "xpu"
        model = self.model_class(config).eval()
        model.to(device).to(memory_format=torch.channels_last)
        input_ids = (
            torch.randint(1, max_token_id, (input_tokens_length,))
            .unsqueeze(0)
            .to(torch.long)
            .to(device)
        )
        generate_kwargs = dict(do_sample=False, temperature=0, num_beams=num_beams)
        ref_res = model.generate(
            input_ids,
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=int(max_new_tokens),
            **generate_kwargs,
        )

        print(
            f"Run model={self.model_class}, num_beams={num_beams}, input_tokens_length={input_tokens_length},"
            f"max_new_tokens={max_new_tokens}, use_static_cache={use_static_cache}"
        )

        model = ipex.llm.optimize(
            model.eval(), dtype=dtype, device=device, inplace=True
        )
        print("Model after optimized: ", model, flush=True)
        if use_static_cache:
            generate_kwargs.update({"cache_implementation": "static"})
        with torch.inference_mode(), torch.no_grad(), torch.xpu.amp.autocast(
            enabled=True, dtype=dtype
        ):
            ipex_res = model.generate(
                input_ids,
                max_new_tokens=int(max_new_tokens),
                min_new_tokens=int(max_new_tokens),
                **generate_kwargs,
            )

        if type(model) in need_recover_models_list.keys():
            reload_module(self.cached_module, model)

        print("ref_res:", ref_res)
        print("ipex_res:", ipex_res)
        max_check_length = input_tokens_length + output_check_len
        self.assertEqual(
            ipex_res[:, input_tokens_length:max_check_length],
            ref_res[:, input_tokens_length:max_check_length],
        )
        return ref_res, ipex_res
