from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers.models.mpt.modeling_mpt import MptForCausalLM

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
        super().get_user_model(config, benchmark)
        if not isinstance(self.model, MptForCausalLM) and not benchmark:
            print(
                "You're using a model from remote hub. To successfully save/load quantized model, \
                please pass configuration file (example: --config-file=model_config/mosaicml_mpt-7b_config.json)."
            )
            exit(0)
        return self.model
