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
        if self.model_backbone in ["GPTJForCausalLM", "PhiForCausalLM"]:
            if not self.distributed:
                self.linear_add_add = _IPEXlinearAddAddCPU(
                    module.linear_add_add.linear, tpp=tpp, woq=woq
                )
            self.linear_gelu = _IPEXlinearNewGeluCPU(
                module.linear_gelu.linear, tpp=tpp, woq=woq
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
        elif self.model_backbone in [
            "WhisperForConditionalGeneration",
            "Phi3ForCausalLM",
            "LlavaLlamaForCausalLM",
            "GitForCausalLM",
            "MixtralForCausalLM",
            "MptForCausalLM",
            "T5ForConditionalGeneration",
            "GPTBigCodeForCausalLM",
            "ChatGLMModel",
            "BloomForCausalLM",
            "FalconForCausalLM",
            "RWForCausalLM",
            "OPTForCausalLM",
            "StableLmForCausalLM",
            "LlamaForCausalLM",
            "MllamaForConditionalGeneration",
            "BaichuanForCausalLM",
            "MistralForCausalLM",
            "QWenLMHeadModel",
            "Qwen2ForCausalLM",
            "YuanForCausalLM",
        ]:
            if not self.distributed:
                if hasattr(module, "linear_add"):
                    self.linear_add = _IPEXlinearAddCPU(
                        module.linear_add.linear, tpp=tpp, woq=woq
                    )
                if hasattr(module, "linear_add_add"):
                    self.linear_add_add = _IPEXlinearAddAddCPU(
                        module.linear_add_add.linear, tpp=tpp, woq=woq
                    )
                if hasattr(module, "mha_linear_add"):
                    self.mha_linear_add = _IPEXlinearAddCPU(
                        module.mha_linear_add.linear, tpp=tpp, woq=woq
                    )
                if hasattr(module, "mlp_linear_add"):
                    self.mlp_linear_add = _IPEXlinearAddCPU(
                        module.mlp_linear_add.linear, tpp=tpp, woq=woq
                    )
                if hasattr(module, "mlp_linear_add_add"):
                    self.mlp_linear_add = _IPEXlinearAddAddCPU(
                        module.mlp_linear_add_add.linear, tpp=tpp, woq=woq
                    )
                if hasattr(module, "encoder_mha_linear_add"):
                    self.encoder_mha_linear_add = _IPEXlinearAddCPU(
                        module.encoder_mha_linear_add.linear, tpp=tpp, woq=woq
                    )
                if hasattr(module, "vision_mha_linear_add"):
                    self.vision_mha_linear_add = _IPEXlinearAddCPU(
                        module.vision_mha_linear_add.linear, tpp=tpp, woq=woq
                    )
                if hasattr(module, "vision_mlp_linear_add"):
                    self.vision_mlp_linear_add = _IPEXlinearAddCPU(
                        module.vision_mlp_linear_add.linear, tpp=tpp, woq=woq
                    )
            if hasattr(module, "linear_gelu"):
                self.linear_gelu = _IPEXlinearGeluCPU(
                    module.linear_gelu.linear, tpp=tpp, woq=woq
                )
            if hasattr(module, "linear_silu_mul"):
                self.linear_silu_mul = _IPEXlinearSiluMulCPU(
                    module.linear_silu_mul.linear_s,
                    module.linear_silu_mul.linear_m,
                    tpp=tpp,
                    woq=woq,
                )
            if hasattr(module, "vision_linear_gelu"):
                self.vision_linear_gelu = _IPEXlinearGeluCPU(
                    module.vision_linear_gelu.linear, tpp=tpp, woq=woq
                )
            if hasattr(module, "linear_mul"):
                self.linear_mul = _IPEXlinearMulCPU(
                    module.linear_mul.linear, tpp=tpp, woq=woq
                )
            if hasattr(module, "linear_relu"):
                self.linear_relu = _IPEXlinearReluCPU(
                    module.linear_relu.linear, tpp=tpp, woq=woq
                )
        else:
            AssertionError(False, "Do not support the optimization of your model yet")
