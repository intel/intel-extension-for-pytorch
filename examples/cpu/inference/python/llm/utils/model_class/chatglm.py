import torch

from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers import AutoModelForCausalLM, AutoTokenizer

import intel_extension_for_pytorch as ipex
import re

class ChatGLMConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "chatglm"
        self.model_id = model_id
        self.to_channels_last = True
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_POS
        self.extra_inputs = (torch.tensor(True),)
        # for smooth quant
        self.default_dataset = "NeelNanda/pile-10k"
        self.use_global_past_key_value = True
        self.use_ipex_autotune = True
    def get_user_model(self, config, benchmark):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float,
            config=config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.config.num_hidden_layers = self.model.config.num_layers
        return self.model

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
