from .llm import LLMConfig, EXAMPLE_INPUTS_MODE

class GPTJConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "gpt-j"
        self.model_id = model_id
        self.to_channels_last = True
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_POS

        # for smooth quant
        self.default_dataset = "NeelNanda/pile-10k"
        self.use_global_past_key_value = True
        self.use_ipex_autotune = True
