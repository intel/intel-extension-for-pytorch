import torch
from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers import T5ForConditionalGeneration
import intel_extension_for_pytorch as ipex

class T5Config(LLMConfig):
    def __init__(self, model_id):
        self.name = "t5"
        self.model_id = model_id
        self.to_channels_last = True
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_ENC

        # for smooth quant
        self.default_dataset = "NeelNanda/pile-10k"
        self.use_global_past_key_value = False
        self.use_ipex_autotune = True

    def get_user_model(self, config, benchmark):
        if benchmark:
            try:
                with ipex.OnDevice(dtype=torch.float, device="meta"):
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float,
                        config=config,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )
            except (RuntimeError, AttributeError):
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float,
                    config=config,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float,
                config=config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        input_ids = torch.ones(32).to(torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        example_inputs = self.model.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask)
        if example_inputs.get("position_ids", None) is not None:
            self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_POS
        return self.model
