import torch

from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers import JambaForCausalLM, AutoTokenizer

import intel_extension_for_pytorch as ipex


class JambaConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "jamba"
        self.model_id = model_id
        self.to_channels_last = False
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_POS_KV

        self.use_global_past_key_value = True
        self.use_ipex_autotune = True

    def get_user_model(self, config, benchmark):
        if benchmark:
            try:
                with ipex.OnDevice(dtype=torch.float, device="meta"):
                    self.model = JambaForCausalLM._from_config(config)
            except (RuntimeError, AttributeError):
                self.model = JambaForCausalLM.from_pretrained(
                    self.model_id,
                    config=config,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.half,
                )
        else:
            self.model = JambaForCausalLM.from_pretrained(
                self.model_id,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float,
            )
        return self.model

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id)
