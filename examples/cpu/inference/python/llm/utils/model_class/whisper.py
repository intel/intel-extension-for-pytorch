import torch
from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers import WhisperForConditionalGeneration, AutoProcessor
import intel_extension_for_pytorch as ipex


class WhisperConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "whisper"
        self.model_id = model_id
        self.to_channels_last = True
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.KV_ENC

        # for smooth quant
        self.default_dataset = "librispeech_asr"
        self.use_global_past_key_value = False
        self.use_ipex_autotune = True

    def get_user_model(self, config, benchmark):
        if benchmark:
            try:
                with ipex.OnDevice(dtype=torch.float, device="meta"):
                    self.model = WhisperForConditionalGeneration.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float,
                        config=config,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )
            except (RuntimeError, AttributeError):
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float,
                    config=config,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float,
                config=config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        return self.model

    def get_tokenizer(self):
        return AutoProcessor.from_pretrained(self.model_id)
