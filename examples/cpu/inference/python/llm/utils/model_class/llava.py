from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
try:
    from llava.model.builder import load_pretrained_model
except ImportError:
    pass

class LlavaConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "llava"
        self.model_id = model_id
        self.to_channels_last = False
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.EMBEDS_MASK_KV

        # for smooth quant
        self.use_global_past_key_value = True
        self.use_ipex_autotune = True
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(self.model_id)
    
    def get_user_model(self, config, benchmark):
        self.model.config = config
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_image_processor(self):
        return self.image_processor
