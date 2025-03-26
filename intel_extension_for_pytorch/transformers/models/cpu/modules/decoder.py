from torch import nn
import torch
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
            "Phi4MMForCausalLM",
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
            "Qwen3ForCausalLM",
            "Qwen2ForCausalLM",
            "YuanForCausalLM",
            "Maira2ForConditionalGeneration",
            "JambaForCausalLM",
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
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
                if hasattr(module, "shared_linear_add_add"):
                    self.shared_linear_add_add = _IPEXlinearAddAddCPU(
                        module.shared_linear_add_add.linear, tpp=tpp, woq=woq
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
            if hasattr(module, "linear_gelus"):
                linear_gelus = [
                    _IPEXlinearGeluCPU(linear_gelu.linear, tpp=tpp, woq=woq)
                    for linear_gelu in module.linear_gelus
                ]
                self.linear_gelus = nn.Sequential(*linear_gelus)
            if hasattr(module, "mlp_linear_silu_mul"):
                self.mlp_linear_silu_mul = _IPEXlinearSiluMulCPU(
                    module.mlp_linear_silu_mul.linear_s,
                    module.mlp_linear_silu_mul.linear_m,
                    tpp=tpp,
                    woq=woq,
                )
            if hasattr(module, "shared_linear_silu_mul"):
                self.shared_linear_silu_mul = _IPEXlinearSiluMulCPU(
                    module.shared_linear_silu_mul.linear_s,
                    module.shared_linear_silu_mul.linear_m,
                    tpp=tpp,
                    woq=woq,
                )
            if self.model_backbone in [
                "DeepseekV2ForCausalLM",
                "DeepseekV3ForCausalLM",
            ]:
                if hasattr(self.mlp, "experts"):
                    # 0: Default, 1: TPP, 2: DNNL, 3: MKL, 4: WOQ
                    self.moe_linear_type = 0
                    if self.mlp.experts[0].gate_proj.weight.dtype in [
                        torch.qint8,
                        torch.int8,
                        torch.uint8,
                        torch.float8_e4m3fn,
                    ]:
                        self.moe_linear_type = 4
                    elif (
                        hasattr(self.mlp.experts[0].gate_proj, "use_tpp")
                        and self.mlp.experts[0].gate_proj.use_tpp
                    ):
                        if not self.mlp.experts[0].gate_proj.tpp_fallback:
                            self.moe_linear_type = 1
                    elif hasattr(self.mlp.experts[0].gate_proj, "use_dnnl"):
                        if self.mlp.experts[0].gate_proj.use_dnnl:
                            self.moe_linear_type = 2
                        else:
                            self.moe_linear_type = 3
                    self.gate_weights = []
                    self.up_weights = []
                    self.down_weights = []
                    self.gate_ctx = []
                    self.up_ctx = []
                    self.down_ctx = []
                    offset = self.mlp.ep_rank * self.mlp.experts_per_rank
                    for expert_idx in range(len(self.mlp.experts)):
                        expert_layer = self.mlp.experts[expert_idx + offset]
                        if self.moe_linear_type in [0, 1]:
                            self.gate_weights.append(expert_layer.gate_proj.weight)
                            self.up_weights.append(expert_layer.up_proj.weight)
                            self.down_weights.append(expert_layer.down_proj.weight)
                        elif self.moe_linear_type in [2, 3]:
                            self.gate_weights.append(
                                expert_layer.gate_proj._get_forward_weight()
                            )
                            self.up_weights.append(
                                expert_layer.up_proj._get_forward_weight()
                            )
                            self.down_weights.append(
                                expert_layer.down_proj._get_forward_weight()
                            )
                            self.gate_ctx.append(expert_layer.gate_proj.ctx)
                            self.up_ctx.append(expert_layer.up_proj.ctx)
                            self.down_ctx.append(expert_layer.down_proj.ctx)
                        else:
                            self.gate_ctx.append(expert_layer.gate_proj._op_context)
                            self.up_ctx.append(expert_layer.up_proj._op_context)
                            self.down_ctx.append(expert_layer.down_proj._op_context)
        else:
            AssertionError(False, "Do not support the optimization of your model yet")


class _IPEXEncoderLayerCPU(nn.Module):
    def __init__(self, module, config, tpp=False, woq=False):
        super().__init__()
        for k, v in module.__dict__.items():
            setattr(self, k, v)
        for k, v in module.__class__.__dict__.items():
            if k.startswith("__"):
                continue
            setattr(self.__class__, k, getattr(module.__class__, k))
        if self.model_backbone in [
            "MllamaForConditionalGeneration",
            "Phi4MMForCausalLM",
        ]:
            if not self.distributed:
                if hasattr(module, "mlp_linear_add"):
                    self.mlp_linear_add = _IPEXlinearAddCPU(
                        module.mlp_linear_add.linear, tpp=tpp, woq=woq
                    )
                if hasattr(module, "mlp_linear_mul"):
                    self.mlp_linear_mul = _IPEXlinearMulCPU(
                        module.mlp_linear_mul.linear, tpp=tpp, woq=woq
                    )
                if hasattr(module, "mha_linear_add"):
                    self.mha_linear_add = _IPEXlinearAddCPU(
                        module.mha_linear_add.linear, tpp=tpp, woq=woq
                    )
            if hasattr(module, "linear_gelu"):
                self.linear_gelu = _IPEXlinearGeluCPU(
                    module.linear_gelu.linear, tpp=tpp, woq=woq
                )
            if hasattr(module, "linear_newgelu"):
                self.linear_newgelu = _IPEXlinearNewGeluCPU(
                    module.linear_newgelu.linear, tpp=tpp, woq=woq
                )
        else:
            AssertionError(False, "Do not support the optimization of your model yet")
