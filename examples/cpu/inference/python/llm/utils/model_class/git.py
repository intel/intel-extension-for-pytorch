from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import intel_extension_for_pytorch as ipex

class GitConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "git"
        self.model_id = model_id
        self.to_channels_last = False
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_PIXEL

        # for smooth quant
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
        return self.model

    def get_tokenizer(self):
        return AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)