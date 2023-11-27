from .llm import LLMConfig, EXAMPLE_INPUTS_MODE

class OPTConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "opt"
        self.model_id = model_id
        self.to_channels_last = True
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.KV_MASK
