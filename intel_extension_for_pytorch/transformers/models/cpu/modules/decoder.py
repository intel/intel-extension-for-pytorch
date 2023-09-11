from torch import nn
import re
from ...cpu.fusions.linear_fusion import (
    _IPEXlinearSiluCPU,
    _IPEXlinearAddCPU,
    _IPEXlinearAddAddCPU,
    _IPEXlinearMulCPU,
    _IPEXlinearGeluCPU,
    _IPEXlinearReluCPU,
)


class _IPEXDecoderLayerCPU(nn.Module):
    def __init__(self, module, config, tpp=False, woq=False):
        super().__init__()
        for k, v in module.__dict__.items():
            setattr(self, k, v)
        for k, v in module.__class__.__dict__.items():
            if k.startswith("__"):
                continue
            setattr(self.__class__, k, getattr(module.__class__, k))
        if re.search("GPTJ", self.model_backbone, re.IGNORECASE):
            if not self.distributed:
                self.linear_add_add = _IPEXlinearAddAddCPU(
                    module.linear_add_add.linear, tpp=tpp, woq=woq
                )
            self.linear_gelu = _IPEXlinearGeluCPU(
                module.linear_gelu.linear, tpp=tpp, woq=woq
            )
        elif re.search("llama", self.model_backbone, re.IGNORECASE):
            if not self.distributed:
                self.mha_linear_add = _IPEXlinearAddCPU(
                    module.mha_linear_add.linear, tpp=tpp, woq=woq
                )
                self.mlp_linear_add = _IPEXlinearAddCPU(
                    module.mlp_linear_add.linear, tpp=tpp, woq=woq
                )
            self.linear_silu = _IPEXlinearSiluCPU(
                module.linear_silu.linear, tpp=tpp, woq=woq
            )
            self.linear_mul = _IPEXlinearMulCPU(
                module.linear_mul.linear, tpp=tpp, woq=woq
            )
        elif re.search("OPT", self.model_backbone, re.IGNORECASE):
            if not self.distributed:
                self.mha_linear_add = _IPEXlinearAddCPU(
                    module.mha_linear_add.linear, tpp=tpp, woq=woq
                )
                self.mlp_linear_add = _IPEXlinearAddCPU(
                    module.mlp_linear_add.linear, tpp=tpp, woq=woq
                )
            self.linear_relu = _IPEXlinearReluCPU(
                module.linear_relu.linear, tpp=tpp, woq=woq
            )
        else:
            AssertionError(False, "Do not support the optimization of your model yet")
