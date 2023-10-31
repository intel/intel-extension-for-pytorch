import torch

from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers.models.falcon.modeling_falcon import FalconForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

import intel_extension_for_pytorch as ipex

model_id_list = ["tiiuae/falcon-40b"]

class FALCONConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "falcon"
        assert model_id in model_id_list, "%s is not a %s model" % (model_id, self.name)
        self.model_id = model_id
        self.to_channels_last = True
        self.trust_remote_code = True
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.KV_MASK
    
    def get_user_model(self, config, benchmark):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float,
            config=config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if not isinstance(self.model, FalconForCausalLM) and not benchmark:
            print(
                "You're using a model from remote hub. To successfully save/load quantized model, \
                please pass configuration file (example: --config-file=model_config/tiiuae_falcon-40b_config.json)."
            )
            exit(0)
        return self.model

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
