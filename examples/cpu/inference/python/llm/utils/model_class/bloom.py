import torch

from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

import intel_extension_for_pytorch as ipex

class BloomConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "bloom"
        self.model_id = model_id
        self.to_channels_last = True
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.KV_MASK
