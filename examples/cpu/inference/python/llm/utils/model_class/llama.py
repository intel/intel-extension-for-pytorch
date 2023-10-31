import torch

from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers import LlamaForCausalLM, LlamaTokenizer

import intel_extension_for_pytorch as ipex

model_id_list = ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf"]

class LLAMAConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "llama"
        assert model_id in model_id_list, "%s is not a %s model" % (model_id, self.name)
        self.model_id = model_id
        self.to_channels_last = False
        self.trust_remote_code = False
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_POS_KV

        # for smooth quant
        self.default_dataset = "NeelNanda/pile-10k"
        self.use_global_past_key_value = True
        self.use_neural_compressor = False
    
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
