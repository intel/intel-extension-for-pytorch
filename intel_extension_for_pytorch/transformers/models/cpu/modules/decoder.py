from torch import nn
from ...cpu.fusions.linear_fusion import (
    _IPEXlinearAddCPU,
    _IPEXlinearAddAddCPU,
    _IPEXlinearNewGeluCPU,
    _IPEXlinearReluCPU,
    _IPEXlinearGeluCPU,
    _IPEXlinearMulCPU,
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
        if self.model_backbone == "GPTJForCausalLM":
            if not self.distributed:
                self.linear_add_add = _IPEXlinearAddAddCPU(
                    module.linear_add_add.linear, tpp=tpp, woq=woq
                )
            self.linear_gelu = _IPEXlinearNewGeluCPU(
                module.linear_gelu.linear, tpp=tpp, woq=woq
            )
        elif self.model_backbone in [
            "LlamaForCausalLM",
            "BaichuanForCausalLM",
            "MistralForCausalLM",
        ]:
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
        elif self.model_backbone == "OPTForCausalLM":
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
        elif (
            self.model_backbone == "FalconForCausalLM"
            or self.model_backbone == "RWForCausalLM"
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
        elif self.model_backbone == "BloomForCausalLM":
            self.linear_gelu = _IPEXlinearGeluCPU(
                module.linear_gelu.linear, tpp=tpp, woq=woq
            )
            if not self.distributed:
                self.linear_add = _IPEXlinearAddCPU(
                    module.linear_add.linear, tpp=tpp, woq=woq
                )
        elif self.model_backbone == "CodeGenForCausalLM":
            if not self.distributed:
                self.linear_add_add = _IPEXlinearAddAddCPU(
                    module.linear_add_add.linear, tpp=tpp, woq=woq
                )
            # woq_linear_gelu has accuracy issues on codegen, disable it
            self.linear_gelu = _IPEXlinearNewGeluCPU(
                module.linear_gelu.linear, tpp=tpp and not woq, woq=False
            )
        elif self.model_backbone == "ChatGLMModel":
            if not self.distributed:
                self.mha_linear_add = _IPEXlinearAddCPU(
                    module.mha_linear_add.linear, tpp=tpp, woq=woq
                )
                self.mlp_linear_add = _IPEXlinearAddCPU(
                    module.mlp_linear_add.linear, tpp=tpp, woq=woq
                )
        elif self.model_backbone == "GPTBigCodeForCausalLM":
            self.linear_gelu = _IPEXlinearGeluCPU(
                module.linear_gelu.linear, tpp=tpp, woq=woq
            )
            if not self.distributed:
                self.mha_linear_add = _IPEXlinearAddCPU(
                    module.mha_linear_add.linear, tpp=tpp, woq=woq
                )
                self.mlp_linear_add = _IPEXlinearAddCPU(
                    module.mlp_linear_add.linear, tpp=tpp, woq=woq
                )
        elif self.model_backbone == "T5ForConditionalGeneration":
            if hasattr(self, "linear_gelu"):
                self.linear_gelu = _IPEXlinearGeluCPU(
                    module.linear_gelu.linear, tpp=tpp, woq=woq
                )
                self.linear_mul = _IPEXlinearMulCPU(
                    module.linear_mul.linear, tpp=tpp, woq=woq
                )
                if not self.distributed:
                    self.linear_add = _IPEXlinearAddCPU(
                        module.linear_add.linear, tpp=tpp, woq=woq
                    )
        elif self.model_backbone == "MptForCausalLM":
            self.linear_gelu = _IPEXlinearGeluCPU(
                module.linear_gelu.linear, tpp=tpp, woq=woq
            )
            if not self.distributed:
                self.linear_add = _IPEXlinearAddCPU(
                    module.linear_add.linear, tpp=tpp, woq=woq
                )
        else:
            AssertionError(False, "Do not support the optimization of your model yet")
