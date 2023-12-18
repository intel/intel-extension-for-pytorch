from .llm import LLMConfig, EXAMPLE_INPUTS_MODE


class MistralConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "mistral"
        self.model_id = model_id
        self.to_channels_last = False
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_POS_KV

        # for smooth quant
        self.use_global_past_key_value = True
        self.use_neural_compressor = False
