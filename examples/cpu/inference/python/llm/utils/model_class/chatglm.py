import torch

from .llm import LLMConfig, EXAMPLE_INPUTS_MODE

class ChatGLMConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "chatglm"
        self.model_id = model_id
        self.to_channels_last = True
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_POS
        self.extra_inputs = (torch.tensor(True),)
        # for smooth quant
        self.default_dataset = "NeelNanda/pile-10k"
        self.use_global_past_key_value = True
        self.use_ipex_autotune = True

    def get_user_model(self, config, benchmark):
        super().get_user_model(config, benchmark)
        self.model.config.num_hidden_layers = self.model.config.num_layers
        return self.model