import torch

from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers import MllamaForConditionalGeneration, AutoProcessor

import intel_extension_for_pytorch as ipex


class MLLAMAConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "mllama"
        self.model_id = model_id
        self.to_channels_last = False
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_POS
        self.use_global_past_key_value = True

    def get_user_model(self, config, benchmark):
        if benchmark:
            try:
                with ipex.OnDevice(dtype=torch.float, device="meta"):
                    self.model = MllamaForConditionalGeneration._from_config(config)
            except (RuntimeError, AttributeError):
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    self.model_id,
                    config=config,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.half,
                )
        else:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_id,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float,
            )
        return self.model

    def get_tokenizer(self):
        return AutoProcessor.from_pretrained(self.model_id)
