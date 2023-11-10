from torch import nn
import re
from ...cpu.fusions.linear_fusion import (
    _IPEXlinearAddCPU,
    _IPEXlinearAddAddCPU,
    _IPEXlinearNewGeluCPU,
    _IPEXlinearReluCPU,
    _IPEXlinearGeluCPU,
    _IPEXlinearSiluMulCPU,
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
            self.linear_gelu = _IPEXlinearNewGeluCPU(
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
            self.linear_silu_mul = _IPEXlinearSiluMulCPU(
                module.linear_silu_mul.linear_s,
                module.linear_silu_mul.linear_m,
                tpp=tpp,
                woq=woq,
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
        elif re.search("falcon", self.model_backbone, re.IGNORECASE) or re.search(
            "rw", self.model_backbone, re.IGNORECASE
        ):
            self.linear_gelu = _IPEXlinearGeluCPU(
                module.linear_gelu.linear, tpp=tpp, woq=woq
            )
            if not self.distributed:
                if hasattr(module, "linear_add_add"):
                    self.linear_add_add = _IPEXlinearAddAddCPU(
                        module.linear_add_add.linear, tpp=tpp, woq=woq
                    )
                elif hasattr(module, "linear_add"):
                    self.linear_add = _IPEXlinearAddCPU(
                        module.linear_add.linear, tpp=tpp, woq=woq
                    )
        elif re.search("codegen", self.model_backbone, re.IGNORECASE):
            if not self.distributed:
                self.linear_add_add = _IPEXlinearAddAddCPU(
                    module.linear_add_add.linear, tpp=tpp, woq=woq
                )
            # woq_linear_gelu has accuracy issues on codegen, disable it
            self.linear_gelu = _IPEXlinearNewGeluCPU(
                module.linear_gelu.linear, tpp=tpp and not woq, woq=False
            )
        else:
            AssertionError(False, "Do not support the optimization of your model yet")
