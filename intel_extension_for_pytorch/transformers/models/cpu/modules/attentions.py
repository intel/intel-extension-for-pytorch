from torch import nn
import re
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

        if not re.search("OPT", self.model_backbone, re.IGNORECASE):
            self._IPEXROPE = _IPEXRopeCPU(
                self.max_position_embeddings,
                self.pos_embd_dim,
                self.rope_base,
                self.model_backbone,
            )

        if re.search("codegen", self.model_backbone, re.IGNORECASE):
            self._IPEXROPE.embed_positions.sin_cos = self.embed_positions
        if re.search("GPTJ", self.model_backbone, re.IGNORECASE) or re.search(
            "LLAMA", self.model_backbone, re.IGNORECASE
        ):
            if hasattr(module, "concat_qkv"):
                self.concat_qkv = _IPEXConcatLinearCPU(
                    module.concat_qkv, tpp=tpp, woq=woq
                )

        self.text_max_length = (
            config.text_max_length if hasattr(config, "text_max_length") else 2048
        )
        self._IPEXScaleDotProduct = _IPEXScaleDotProductCPU(
            text_max_length=self.text_max_length
        )
