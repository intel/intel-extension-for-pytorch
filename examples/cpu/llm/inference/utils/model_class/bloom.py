import torch  # noqa F401

from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers.models.bloom.modeling_bloom import BloomForCausalLM  # noqa F401
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa F401

import intel_extension_for_pytorch as ipex  # noqa F401


class BloomConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "bloom"
        self.model_id = model_id
        self.to_channels_last = True
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV
        # for smooth quant
        self.default_dataset = "NeelNanda/pile-10k"
        self.use_global_past_key_value = True
        self.use_ipex_autotune = True
