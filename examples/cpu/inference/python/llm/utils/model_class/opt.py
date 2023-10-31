from .llm import LLMConfig, EXAMPLE_INPUTS_MODE

model_id_list = ["facebook/opt-30b", "facebook/opt-1.3b"]

class OPTConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "opt"
        assert model_id in model_id_list, "%s is not a %s model" % (model_id, self.name)
        self.model_id = model_id
        self.to_channels_last = True
        self.trust_remote_code = False
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV
