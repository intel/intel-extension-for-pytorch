import torch

from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers import AutoModelForCausalLM, AutoTokenizer

import intel_extension_for_pytorch as ipex
import re

class BaichuanConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "baichuan"
        if re.search("baichuan2", model_id.lower()):
            self.name = "baichuan2"
        self.model_id = model_id
        self.to_channels_last = True
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV
        # for smooth quant
        self.default_dataset = "NeelNanda/pile-10k"
        self.use_global_past_key_value = True
        self.use_ipex_autotune = True
    def get_user_model(self, config, benchmark):
        if benchmark:
            try:
                with ipex.OnDevice(dtype=torch.float, device="meta"):
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float,
                        config=config,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )
            except (RuntimeError, AttributeError):
                self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float,
                        config=config,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float,
                        config=config,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )
        input_ids = torch.ones(32).to(torch.long)
        example_inputs = self.model.prepare_inputs_for_generation(input_ids)
        if "position_ids" in example_inputs:
            self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_POS
        return self.model

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
