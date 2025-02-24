from .llm import LLMConfig, EXAMPLE_INPUTS_MODE
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import intel_extension_for_pytorch as ipex  # noqa F401


class PhiConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "phi"
        self.model_id = model_id
        self.to_channels_last = True
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_POS

        # for smooth quant
        self.default_dataset = "NeelNanda/pile-10k"
        self.use_global_past_key_value = True
        self.use_ipex_autotune = True


class Phi3Config(LLMConfig):
    def __init__(self, model_id):
        self.name = "phi-3"
        self.model_id = model_id
        self.to_channels_last = True
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_POS

        # for smooth quant
        self.default_dataset = "NeelNanda/pile-10k"
        self.use_global_past_key_value = True
        self.use_ipex_autotune = True


class PhiOConfig(LLMConfig):
    def __init__(self, model_id):
        self.name = "phio"
        self.model_id = model_id
        self.to_channels_last = True
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_KV_POS

        # for smooth quant
        self.default_dataset = "NeelNanda/pile-10k"
        self.use_global_past_key_value = True
        self.use_ipex_autotune = True

    def get_user_model(self, config, load_to_meta_device):
        if load_to_meta_device:
            try:
                with ipex.OnDevice(dtype=torch.float, device="meta"):
                    self.model = AutoModelForCausalLM.from_config(
                        config, trust_remote_code=True, attn_implementation="eager"
                    )
            except (RuntimeError, AttributeError):
                self.model = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=True, attn_implementation="eager"
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float,
                    config=config,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    attn_implementation="eager",
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float,
                config=config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager",
            )
        return self.model

    def get_tokenizer(self):
        return AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
