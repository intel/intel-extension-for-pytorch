from torch import nn
from ...cpu.fusions.mha_fusion import (
    _IPEXRopeCPU,
    _IPEXScaleDotProductCPU,
)
from ...cpu.fusions.linear_fusion import (
    _IPEXConcatLinearCPU,
)


class _IPEXAttentionCPU(nn.Module):
    def __init__(self, module, config, tpp=False, woq=False):
        super().__init__()
        for k, v in module.__dict__.items():
            setattr(self, k, v)
        for k, v in module.__class__.__dict__.items():
            if k.startswith("__"):
                continue
            setattr(self.__class__, k, getattr(module.__class__, k))

        if (
            self.model_backbone
            not in [
                "OPTForCausalLM",
                "BloomForCausalLM",
                "T5ForConditionalGeneration",
                "MptForCausalLM",
            ]
            or self.model_backbone == "BaichuanForCausalLM"
            and hasattr(module, "rotary_emb")
        ):
            self._IPEXROPE = _IPEXRopeCPU(
                self.max_position_embeddings,
                self.pos_embd_dim,
                self.rope_base,
                self.model_backbone,
            )
        if self.model_backbone in [
            "GPTJForCausalLM",
            "LlamaForCausalLM",
            "MistralForCausalLM",
        ]:
            if hasattr(module, "concat_qkv"):
                self.concat_qkv = _IPEXConcatLinearCPU(
                    module.concat_qkv, tpp=tpp, woq=woq
                )

        if self.model_backbone in ["CodeGenForCausalLM"]:
            self._IPEXROPE.embed_positions.sin_cos = self.embed_positions
        self.text_max_length = (
            config.text_max_length if hasattr(config, "text_max_length") else 2048
        )
        self._IPEXScaleDotProduct = _IPEXScaleDotProductCPU(
            text_max_length=self.text_max_length
        )
