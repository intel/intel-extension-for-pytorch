from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers.models.mpt.modeling_mpt import MptForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class MPTConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "mpt"
        self.model_id = model_id
        self.to_channels_last = False
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV

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
        if not isinstance(self.model, MptForCausalLM) and not benchmark:
            print(
                "You're using a model from remote hub. To successfully save/load quantized model, \
                please pass configuration file (example: --config-file=model_config/mosaicml_mpt-7b_config.json)."
            )
            exit(0)
        return self.model

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
