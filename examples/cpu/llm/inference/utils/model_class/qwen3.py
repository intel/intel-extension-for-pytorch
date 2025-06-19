from .llm import LLMConfig, EXAMPLE_INPUTS_MODE


class Qwen3Config(LLMConfig):
    def __init__(self, model_id):
        self.name = "qwen3"
        self.model_id = model_id
        self.to_channels_last = False
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_POS

        # for smooth quant
        self.use_global_past_key_value = True
        self.use_ipex_autotune = True
