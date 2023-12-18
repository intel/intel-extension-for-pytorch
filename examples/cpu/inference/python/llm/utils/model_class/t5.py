import torch
from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers import T5ForConditionalGeneration

class T5Config(LLMConfig):
    def __init__(self, model_id):
        self.name = "t5"
        self.model_id = model_id
        self.to_channels_last = True
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_ENC

    def get_user_model(self, config, benchmark):
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float,
            config=config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        input_ids = torch.ones(32).to(torch.long)
        example_inputs = self.model.prepare_inputs_for_generation(input_ids)
        if "position_ids" in example_inputs:
            self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_POS
        return self.model
