from .llm import LLMConfig, EXAMPLE_INPUTS_MODE

import torch
from transformers import AutoModelForCausalLM

import intel_extension_for_pytorch as ipex


class DeepseekV2Config(LLMConfig):
    def __init__(self, model_id):
        self.name = "deepseekv2"
        self.model_id = model_id
        self.to_channels_last = False
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_POS_KV

        self.use_global_past_key_value = True
        self.use_ipex_autotune = True

    def get_user_model(self, config, benchmark):
        if benchmark:
            try:
                with ipex.OnDevice(dtype=torch.float, device="meta"):
                    self.model = AutoModelForCausalLM.from_config(
                        config, trust_remote_code=True
                    )
            except (RuntimeError, AttributeError):
                self.model = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=True
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    config=config,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                config=config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        return self.model


class DeepseekV3Config(LLMConfig):
    def __init__(self, model_id):
        self.name = "deepseekv3"
        self.model_id = model_id
        self.to_channels_last = False
        self.example_inputs_mode = EXAMPLE_INPUTS_MODE.MASK_POS_KV

        self.use_global_past_key_value = True
        self.use_ipex_autotune = True

    def get_user_model(self, config, benchmark):
        if benchmark:
            try:
                with ipex.OnDevice(dtype=torch.float, device="meta"):
                    self.model = AutoModelForCausalLM.from_config(
                        config, trust_remote_code=True
                    )
            except (RuntimeError, AttributeError):
                self.model = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=True
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    config=config,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                config=config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        return self.model
