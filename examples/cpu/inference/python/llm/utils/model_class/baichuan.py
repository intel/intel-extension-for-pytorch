import torch

from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
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
        super().get_user_model(config, benchmark)
        input_ids = torch.ones(32).to(torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        example_inputs = self.model.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask)
        if example_inputs.get("position_ids", None) is not None:
            self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_POS
        return self.model
