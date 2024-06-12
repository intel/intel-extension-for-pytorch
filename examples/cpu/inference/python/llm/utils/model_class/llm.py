from abc import ABC, abstractmethod
from enum import IntEnum

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import intel_extension_for_pytorch as ipex

class EXAMPLE_INPUTS_MODE(IntEnum):
    MASK_KV = 1
    KV_MASK = 2
    MASK_POS_KV = 3
    MASK_KV_POS = 4
    MASK_KV_ENC = 5
    MASK_KV_PIXEL = 6
    EMBEDS_MASK_KV = 7

class LLMConfig(ABC):
    @abstractmethod
    def __init__(self, model_id):
        '''
            self.name: model name
            self.model_id: model id
            self.to_channels_last: channels last model
            self.example_inputs_mode:
                MASK_KV: input_ids+attn_mask+past_kv
                KV_MASK: input_ids+past_kv+attn_mask
                MASK_POS_KV: input_ids+attn_mask+position_ids+past_kv
                MASK_KV_POS: input_ids+attn_mask+past_kv+position_ids
                MASK_KV_ENC: input_ids+attn_mask+past_kv+encoder_output

            # if support smooth quant
            self.default_dataset: default dataset
            self.use_global_past_key_value:
                use_global_past_key_value in collate_batch
            self.use_ipex_autotune:
                use_ipex_autotune in ipex_smooth_quant
        '''
        self.model_id = model_id

    def get_user_model(self, config, benchmark):
        if benchmark:
            try:
                with ipex.OnDevice(dtype=torch.float, device="meta"):
                    self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            except (RuntimeError, AttributeError):
                self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.float, config=config, low_cpu_mem_usage=True, trust_remote_code=True
            )
        return self.model

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
