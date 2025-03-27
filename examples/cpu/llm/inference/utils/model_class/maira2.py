import torch

from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers import AutoModelForCausalLM, AutoProcessor

import intel_extension_for_pytorch as ipex


class MAIRA2Config(LLMConfig):
    def __init__(self, model_id):
        self.name = "maira2"
        self.model_id = model_id
        self.to_channels_last = False
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_POS_KV
        self.use_global_past_key_value = True

    def get_user_model(self, config, load_to_meta_device):
        if load_to_meta_device:
            try:
                with ipex.OnDevice(dtype=torch.float, device="meta"):
                    self.model = AutoModelForCausalLM._from_config(
                        config, trust_remote_code=True
                    )
            except (RuntimeError, AttributeError):
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        config=config,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        torch_dtype=torch.half,
                    )
                except NotImplementedError:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        trust_remote_code=True,
                        config=config,
                        torch_dtype=torch.float,
                    )
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    config=config,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    torch_dtype=torch.float,
                )
            except NotImplementedError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    config=config,
                    torch_dtype=torch.float,
                )
        return self.model

    def get_tokenizer(self):
        return AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
