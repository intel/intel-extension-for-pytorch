import torch

from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers import LlamaForCausalLM, LlamaTokenizer

import intel_extension_for_pytorch as ipex

class LLAMAConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "llama"
        self.model_id = model_id
        self.to_channels_last = False
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_POS

        # for smooth quant
        self.default_dataset = "NeelNanda/pile-10k"
        self.use_global_past_key_value = True
        self.use_ipex_autotune = True
    
    def get_user_model(self, config, benchmark):
        if benchmark:
            try:
                with ipex.OnDevice(dtype=torch.float, device="meta"):
                    self.model = LlamaForCausalLM._from_config(config)
            except (RuntimeError, AttributeError):
                self.model = LlamaForCausalLM.from_pretrained(
                    self.model_id, config=config, low_cpu_mem_usage=True, torch_dtype=torch.half
                )
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_id, config=config, low_cpu_mem_usage=True, torch_dtype=torch.float
            )
        return self.model

    def get_tokenizer(self):
        return LlamaTokenizer.from_pretrained(self.model_id)
