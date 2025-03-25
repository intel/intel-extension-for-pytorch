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


def woq_quant_and_pack(weight, group_size):
    from intel_extension_for_pytorch.quantization import (
        WoqWeightDtype,
        WoqLowpMode,
        WoqActQuantMode,
        quantize_per_channel,
        quantize_per_block,
    )

    dtype = WoqWeightDtype.INT8
    assert group_size == -1, "current fused MOE WOQ only support group size = -1"
    if group_size == -1:
        qweight, scales, zero_points = quantize_per_channel(
            weight, dtype, None, None, False
        )
    else:
        qweight, scales, zero_points = quantize_per_block(
            weight, dtype, group_size, None, None, False
        )

    _op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack(
        qweight,
        dtype,
        [weight.shape[0], weight.shape[1]],
        scales,
        zero_points,
        None,  # bias
        None,  # g_idx
        None,  # batch size
        group_size,
        WoqLowpMode.BF16,  # lowp-mode
        WoqActQuantMode.NONE,  # act_quant_mode
        False,  # cache_weight_for_large_batch
    )
    # qweight: {N/block_n, K/block_k, block_k, block_n}
    return (
        _op_context.get_weight(),
        _op_context.get_scales(),
        _op_context.get_zero_points(),
    )

def woq_pack(plain_qweight, plain_scales, plain_zp=None, group_size=-1):
    from intel_extension_for_pytorch.quantization import (
        WoqWeightDtype,
        WoqLowpMode,
        WoqActQuantMode,
    )

    dtype = WoqWeightDtype.INT8
    assert group_size == -1, "current fused MOE WOQ only support group size = -1"

    _op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack(
        plain_qweight,
        dtype,
        [plain_qweight.shape[0], plain_qweight.shape[1]],
        plain_scales,
        plain_zp,
        None,  # bias
        None,  # g_idx
        None,  # batch size
        group_size,
        WoqLowpMode.BF16,  # lowp-mode
        WoqActQuantMode.NONE,  # act_quant_mode
        False,  # cache_weight_for_large_batch
    )
    # pack_qweight: {N/block_n, K/block_k, block_k, block_n}
    return (
        _op_context.get_weight(),
        _op_context.get_scales(),
        _op_context.get_zero_points(),
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
                self.unify_experts = False
                self.fuse_moe_woq_sym = False
                if hasattr(self.mlp, "shared_experts"):
                    if config.n_shared_experts == 1:
                        self.unify_experts = True
                        self.unify_shared_expert_id = config.n_routed_experts + 1
                        print("[INFO] using unify_experts for shared MOE...")
                    if hasattr(self, "deepseek_lowbit_load") and self.deepseek_lowbit_load:
                        w13_shared_qweight_list = []
                        w13_shared_scale_list = []
                        w13_shared_zp_list = []
                        w2_shared_qweight_list = []
                        w2_shared_scale_list = []
                        w2_shared_zp_list = []
                        group_size = -1
                        dtype = torch.bfloat16
                        self.fuse_moe_woq_sym = True
                        assert module.mlp.shared_experts.gate_proj._group_size == -1, "Block wise WOQ fusedMoE is not supported yet..."
                        assert module.mlp.shared_experts.gate_proj._op_context.get_zero_points() is None, "Expecting WOQ fusedMoE with sym quant, no zp..."
                        shared_moe_gate_proj_weight = module.mlp.shared_experts.gate_proj._op_context.to_public(
                            module.mlp.shared_experts.gate_proj._op_context.get_weight()
                        )
                        shared_moe_gate_proj_scale = module.mlp.shared_experts.gate_proj._op_context.get_scales()


                        shared_moe_up_proj_weight = module.mlp.shared_experts.up_proj._op_context.to_public(
                            module.mlp.shared_experts.up_proj._op_context.get_weight()
                        )
                        shared_moe_up_proj_scale = module.mlp.shared_experts.up_proj._op_context.get_scales()

                        shared_weights_list = [
                            shared_moe_gate_proj_weight,
                            shared_moe_up_proj_weight,
                        ]
                        shared_scale_list = [
                            shared_moe_gate_proj_scale,
                            shared_moe_up_proj_scale,
                        ]

                        concat_shared_weight = torch.concat(shared_weights_list, 0)
                        concat_shared_scale = torch.concat(shared_scale_list, 0)

                        del self.__dict__["_modules"]["mlp"].shared_experts.gate_proj
                        del self.__dict__["_modules"]["mlp"].shared_experts.up_proj
                        pack_shared_weight, pack_shared_scale, _ = woq_pack(concat_shared_weight, concat_shared_scale)
                        w13_shared_qweight_list.append(pack_shared_weight)
                        w13_shared_scale_list.append(pack_shared_scale)
                        w13_shared_zp_list.append(torch.tensor(0).to(dtype))

                        shared_moe_down_proj = module.mlp.shared_experts.down_proj._op_context.get_weight()
                        shared_moe_down_proj_scale = module.mlp.shared_experts.down_proj._op_context.get_scales()

                        w2_shared_qweight_list.append(shared_moe_down_proj)
                        w2_shared_scale_list.append(shared_moe_down_proj_scale)
                        w2_shared_zp_list.append(torch.tensor(0).to(dtype))
                        del self.__dict__["_modules"]["mlp"].shared_experts.down_proj

                        self.w13_shared_weight = torch.stack(w13_shared_qweight_list).detach()
                        self.w13_shared_scale = (
                            torch.stack(w13_shared_scale_list).detach().to(dtype)
                        )
                        self.w13_shared_zp = (
                            torch.stack(w13_shared_zp_list).detach().to(dtype)
                        )
                        self.w2_shared_weight = torch.stack(w2_shared_qweight_list).detach()
                        self.w2_shared_scale = (
                            torch.stack(w2_shared_scale_list).detach().to(dtype)
                        )
                        self.w2_shared_zp = torch.stack(w2_shared_zp_list).detach().to(dtype)

                        print("[INFO] Using fused shared MOE WOQ INT8 lowbit weights path...")
                    else:
                        if (
                            self.use_fused_moe or self.use_fused_moe_woq
                        ) and self.mlp.shared_experts.w13_shared_weight.device.type != "meta":
                            dtype = self.mlp.shared_experts.w13_shared_weight.dtype
                            if not self.use_fused_moe_woq:
                                w13_shared_weight = torch.stack(
                                    [
                                        self.mlp.shared_experts.w13_shared_weight
                                    ]
                                ).detach()
                                self.w13_shared_weight = (
                                    torch.ops.torch_ipex.convert_weight_packed_bf16(w13_shared_weight)
                                )
                                del self.mlp.shared_experts.w13_shared_weight
                                w2_shared_weight = torch.stack(
                                    [
                                        self.mlp.shared_experts.w2_shared_weight
                                    ]
                                ).detach()
                                self.w2_shared_weight = torch.ops.torch_ipex.convert_weight_packed_bf16(
                                    w2_shared_weight
                                )
                                del self.mlp.shared_experts.w2_shared_weight
                                # dummy scale/zps
                                self.w13_shared_scale = torch.tensor(0).to(dtype)
                                self.w13_shared_zp = torch.tensor(0).to(dtype)
                                self.w2_shared_scale = torch.tensor(0).to(dtype)
                                self.w2_shared_zp = torch.tensor(0).to(dtype)
                                print("[INFO] Using fused shared MOE bf16 path...")
                            else:
                                w13_shared_qweight_list = []
                                w13_shared_scale_list = []
                                w13_shared_zp_list = []
                                w2_shared_qweight_list = []
                                w2_shared_scale_list = []
                                w2_shared_zp_list = []
                                group_size = -1
                                w13_shared_qweight, w13_shared_scale, w13_shared_zp = woq_quant_and_pack(
                                    self.mlp.shared_experts.w13_shared_weight, group_size
                                )
                                del self.mlp.shared_experts.w13_shared_weight
                                w13_shared_qweight_list.append(w13_shared_qweight)
                                w13_shared_scale_list.append(w13_shared_scale)
                                w13_shared_zp_list.append(w13_shared_zp)
                                w2_shared_qweight, w2_shared_scale, w2_shared_zp = woq_quant_and_pack(
                                    self.mlp.shared_experts.w2_shared_weight, group_size
                                )
                                del self.mlp.shared_experts.w2_shared_weight

                                w2_shared_qweight_list.append(w2_shared_qweight)
                                w2_shared_scale_list.append(w2_shared_scale)
                                w2_shared_zp_list.append(w2_shared_zp)
                                self.w13_shared_weight = torch.stack(w13_shared_qweight_list).detach()
                                self.w13_shared_scale = (
                                    torch.stack(w13_shared_scale_list).detach().to(dtype)
                                )
                                self.w13_shared_zp = (
                                    torch.stack(w13_shared_zp_list).detach().to(dtype)
                                )
                                self.w2_shared_weight = torch.stack(w2_shared_qweight_list).detach()
                                self.w2_shared_scale = (
                                    torch.stack(w2_shared_scale_list).detach().to(dtype)
                                )
                                self.w2_shared_zp = torch.stack(w2_shared_zp_list).detach().to(dtype)

                                print("[INFO] Using fused shared MOE WOQ INT8 path...")

                if hasattr(self.mlp, "experts"):
                    if hasattr(self, "deepseek_lowbit_load") and self.deepseek_lowbit_load:
                        w13_qweight_list = []
                        w13_scale_list = []
                        w13_zp_list = []
                        w2_qweight_list = []
                        w2_scale_list = []
                        w2_zp_list = []
                        group_size = -1
                        dtype = torch.bfloat16

                        assert module.mlp.experts[0].gate_proj._group_size == -1, "Block wise WOQ fusedMoE is not supported yet..."
                        assert module.mlp.experts[0].gate_proj._op_context.get_zero_points() is None, "Expecting WOQ fusedMoE with sym quant, no zp..."
                        self.fuse_moe_woq_sym = True
                        for idx in range(config.n_routed_experts):
                            moe_gate_proj_weight = module.mlp.experts[idx].gate_proj._op_context.to_public(
                                module.mlp.experts[idx].gate_proj._op_context.get_weight()
                            )
                            moe_gate_proj_scale = module.mlp.experts[idx].gate_proj._op_context.get_scales()

                            moe_up_proj_weight = module.mlp.experts[idx].up_proj._op_context.to_public(
                                module.mlp.experts[idx].up_proj._op_context.get_weight()
                            )
                            moe_up_proj_scale = module.mlp.experts[idx].up_proj._op_context.get_scales()

                            weights_list = [
                                moe_gate_proj_weight,
                                moe_up_proj_weight,
                            ]
                            scale_list = [
                                moe_gate_proj_scale,
                                moe_up_proj_scale,
                            ]

                            concat_weight = torch.concat(weights_list, 0)
                            concat_scale = torch.concat(scale_list, 0)
                            del self.__dict__["_modules"]["mlp"].experts[idx].gate_proj
                            del self.__dict__["_modules"]["mlp"].experts[idx].up_proj
                            pack_weight, pack_scale, _ = woq_pack(concat_weight, concat_scale)
                            w13_qweight_list.append(pack_weight)
                            w13_scale_list.append(pack_scale)
                            w13_zp_list.append(torch.tensor(0).to(dtype))

                            moe_down_proj_weight = module.mlp.experts[idx].down_proj._op_context.get_weight()
                            moe_down_proj_scale = module.mlp.experts[idx].down_proj._op_context.get_scales()
                            w2_qweight_list.append(moe_down_proj_weight)
                            w2_scale_list.append(moe_down_proj_scale)
                            w2_zp_list.append(torch.tensor(0).to(dtype))
                            del self.__dict__["_modules"]["mlp"].experts[idx].down_proj
                        if self.unify_experts:
                            w13_qweight_list.append(self.w13_shared_weight[0])
                            del self.w13_shared_weight
                            w13_scale_list.append(self.w13_shared_scale[0])
                            del self.w13_shared_scale
                            w13_zp_list.append(self.w13_shared_zp[0])
                            del self.w13_shared_zp
                            w2_qweight_list.append(self.w2_shared_weight[0])
                            del self.w2_shared_weight
                            w2_scale_list.append(self.w2_shared_scale[0])
                            del self.w2_shared_scale
                            w2_zp_list.append(self.w2_shared_zp[0])
                            del self.w2_shared_zp
                        self.w13_weight = torch.stack(w13_qweight_list).detach()
                        self.w13_scale = (
                            torch.stack(w13_scale_list).detach().to(dtype)
                        )
                        self.w13_zp = (
                            torch.stack(w13_zp_list).detach().to(dtype)
                        )
                        self.w2_weight = torch.stack(w2_qweight_list).detach()
                        self.w2_scale = (
                            torch.stack(w2_scale_list).detach().to(dtype)
                        )
                        self.w2_zp = torch.stack(w2_zp_list).detach().to(dtype)

                        print("[INFO] Using fused MOE WOQ INT8 lowbit weights path...")
                    else:
                        if (
                            self.use_fused_moe or self.use_fused_moe_woq
                        ) and self.mlp.experts[0].w13_weight.device.type != "meta":
                            dtype = self.mlp.experts[0].w13_weight.dtype
                            if not self.use_fused_moe_woq:
                                self.unify_experts = False
                                w13_weight = torch.stack(
                                    [
                                        self.mlp.experts[idx].w13_weight
                                        for idx in range(len(self.mlp.experts))
                                    ]
                                ).detach()
                                self.w13_weight = (
                                    torch.ops.torch_ipex.convert_weight_packed_bf16(w13_weight)
                                )
                                for idx in range(len(self.mlp.experts)):
                                    del self.mlp.experts[idx].w13_weight
                                w2_weight = torch.stack(
                                    [
                                        self.mlp.experts[idx].w2_weight
                                        for idx in range(len(self.mlp.experts))
                                    ]
                                ).detach()
                                self.w2_weight = torch.ops.torch_ipex.convert_weight_packed_bf16(
                                    w2_weight
                                )
                                for idx in range(len(self.mlp.experts)):
                                    del self.mlp.experts[idx].w2_weight
                                # dummy scale/zps

                                self.w13_scale = torch.tensor(0).to(dtype)
                                self.w13_zp = torch.tensor(0).to(dtype)
                                self.w2_scale = torch.tensor(0).to(dtype)
                                self.w2_zp = torch.tensor(0).to(dtype)
                                print("[INFO] Using fused MOE bf16 path...")
                            else:
                                w13_qweight_list = []
                                w13_scale_list = []
                                w13_zp_list = []
                                w2_qweight_list = []
                                w2_scale_list = []
                                w2_zp_list = []
                                group_size = -1
                                for idx in range(len(self.mlp.experts)):
                                    w13_qweight, w13_scale, w13_zp = woq_quant_and_pack(
                                        self.mlp.experts[idx].w13_weight, group_size
                                    )
                                    del self.mlp.experts[idx].w13_weight
                                    w13_qweight_list.append(w13_qweight)
                                    w13_scale_list.append(w13_scale)
                                    w13_zp_list.append(w13_zp)
                                    w2_qweight, w2_scale, w2_zp = woq_quant_and_pack(
                                        self.mlp.experts[idx].w2_weight, group_size
                                    )
                                    del self.mlp.experts[idx].w2_weight

                                    w2_qweight_list.append(w2_qweight)
                                    w2_scale_list.append(w2_scale)
                                    w2_zp_list.append(w2_zp)
                                if self.unify_experts:
                                    w13_qweight_list.append(self.w13_shared_weight[0])
                                    del self.w13_shared_weight
                                    w13_scale_list.append(self.w13_shared_scale[0])
                                    del self.w13_shared_scale
                                    w13_zp_list.append(self.w13_shared_zp[0])
                                    del self.w13_shared_zp
                                    w2_qweight_list.append(self.w2_shared_weight[0])
                                    del self.w2_shared_weight
                                    w2_scale_list.append(self.w2_shared_scale[0])
                                    del self.w2_shared_scale
                                    w2_zp_list.append(self.w2_shared_zp[0])
                                    del self.w2_shared_zp
                                self.w13_weight = torch.stack(w13_qweight_list).detach()
                                self.w13_scale = (
                                    torch.stack(w13_scale_list).detach().to(dtype)
                                )
                                self.w13_zp = (
                                    torch.stack(w13_zp_list).detach().to(dtype)
                                )
                                self.w2_weight = torch.stack(w2_qweight_list).detach()
                                self.w2_scale = (
                                    torch.stack(w2_scale_list).detach().to(dtype)
                                )
                                self.w2_zp = torch.stack(w2_zp_list).detach().to(dtype)

                                print("[INFO] Using fused MOE WOQ INT8 path...")
                        elif not (self.use_fused_moe or self.use_fused_moe_woq):
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
            if hasattr(module, "linear_gelu"):
                self.linear_silu = _IPEXlinearGeluCPU(
                    module.linear_gelu.linear, tpp=tpp, woq=woq
                )
        else:
            AssertionError(False, "Do not support the optimization of your model yet")
