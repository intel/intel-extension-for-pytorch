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
from intel_extension_for_pytorch.quantization import (
    WoqWeightQScheme,
    WoqWeightDtype,
    WoqLowpMode,
)


def woq_quant_and_pack(weight, group_size, dtype, lowp_mode, sym_quant_weight):
    from intel_extension_for_pytorch.quantization import (
        WoqWeightDtype,
        WoqLowpMode,
        WoqActQuantMode,
        quantize_per_channel,
        quantize_per_block,
    )

    if group_size == -1:
        qweight, scales, zero_points = quantize_per_channel(
            weight, dtype, None, None, sym_quant_weight
        )
    else:
        qweight, scales, zero_points = quantize_per_block(
            weight, dtype, group_size, None, None, sym_quant_weight
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
        lowp_mode,
        WoqActQuantMode.NONE,  # act_quant_mode
        False,  # cache_weight_for_large_batch
    )
    # qweight: {N/block_n, K/block_k, block_k, block_n}
    if (
        dtype == WoqWeightDtype.INT8
        and lowp_mode == WoqLowpMode.INT8
        and _op_context.get_weight().dim() == 4
    ):
        n_blocks, k_blocks, block_k, block_n = _op_context.get_weight().shape
        weight_view = qweight.view([n_blocks, block_n, k_blocks, block_k])
        compensation = torch.sum(weight_view, dim=-1, keepdim=False, dtype=torch.int32)
        compensation = compensation.permute([0, 2, 1]).contiguous()
    else:
        compensation = None
    qweight = _op_context.get_weight()
    scale = _op_context.get_scales()
    zero_point = _op_context.get_zero_points()
    return (qweight, scale, zero_point, compensation)


def woq_pack(plain_qweight, plain_scales, plain_zp, group_size, dtype, lowp_mode):
    from intel_extension_for_pytorch.quantization import (
        WoqWeightDtype,
        WoqLowpMode,
        WoqActQuantMode,
    )

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
        lowp_mode,
        WoqActQuantMode.NONE,  # act_quant_mode
        False,  # cache_weight_for_large_batch
    )
    if (
        dtype == WoqWeightDtype.INT8
        and lowp_mode == WoqLowpMode.INT8
        and _op_context.get_weight().dim() == 4
    ):
        n_blocks, k_blocks, block_k, block_n = _op_context.get_weight().shape
        weight_view = plain_qweight.view([n_blocks, block_n, k_blocks, block_k])
        compensation = torch.sum(weight_view, dim=-1, keepdim=False, dtype=torch.int32)
        compensation = compensation.permute([0, 2, 1]).contiguous()
    else:
        compensation = None
    # pack_qweight: {N/block_n, K/block_k, block_k, block_n}
    return (
        _op_context.get_weight(),
        _op_context.get_scales(),
        _op_context.get_zero_points(),
        compensation,
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
            "Qwen3MoeForCausalLM",
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
                "Qwen3MoeForCausalLM",
            ]:
                self.unify_experts = False
                if hasattr(self.mlp, "shared_experts"):
                    if config.n_shared_experts == 1:
                        self.unify_experts = True
                        self.unify_shared_expert_id = config.n_routed_experts + 1
                        print("[INFO] using unify_experts for shared MOE...")
                    if (
                        hasattr(self, "deepseek_lowbit_load")
                        and self.deepseek_lowbit_load
                    ):
                        w13_shared_qweight_list = []
                        w13_shared_scale_list = []
                        w13_shared_zp_list = []
                        w13_shared_compensation_list = []
                        w2_shared_qweight_list = []
                        w2_shared_scale_list = []
                        w2_shared_zp_list = []
                        w2_shared_compensation_list = []
                        dtype = torch.bfloat16
                        self.woq_weight_dtype = module.mlp.experts[0].gate_proj.dtype
                        self.woq_group_size = module.mlp.experts[
                            0
                        ].gate_proj._group_size
                        self.woq_lowp_mode = module.mlp.experts[0].gate_proj._lowp_mode
                        self.woq_weight_qscheme = module.mlp.experts[
                            0
                        ].gate_proj._weight_qscheme
                        is_da8w8 = (
                            self.woq_weight_dtype == WoqWeightDtype.INT8
                            and self.woq_lowp_mode == WoqLowpMode.INT8
                        )
                        assert (
                            self.woq_group_size == -1
                        ), "Block wise WOQ fusedMoE is not supported yet..."
                        shared_moe_gate_proj_weight = module.mlp.shared_experts.gate_proj._op_context.to_public(
                            module.mlp.shared_experts.gate_proj._op_context.get_weight()
                        )
                        shared_moe_gate_proj_scale = (
                            module.mlp.shared_experts.gate_proj._op_context.get_scales()
                        )
                        shared_moe_gate_proj_zp = (
                            module.mlp.shared_experts.gate_proj._op_context.get_zero_points()
                        )

                        shared_moe_up_proj_weight = module.mlp.shared_experts.up_proj._op_context.to_public(
                            module.mlp.shared_experts.up_proj._op_context.get_weight()
                        )
                        shared_moe_up_proj_scale = (
                            module.mlp.shared_experts.up_proj._op_context.get_scales()
                        )
                        shared_moe_up_proj_zp = (
                            module.mlp.shared_experts.up_proj._op_context.get_zero_points()
                        )

                        shared_weights_list = [
                            shared_moe_gate_proj_weight,
                            shared_moe_up_proj_weight,
                        ]
                        shared_scale_list = [
                            shared_moe_gate_proj_scale,
                            shared_moe_up_proj_scale,
                        ]
                        shared_zp_list = []
                        if shared_moe_gate_proj_zp is not None:
                            shared_zp_list.append(shared_moe_gate_proj_zp)
                        if shared_moe_up_proj_zp is not None:
                            shared_zp_list.append(shared_moe_up_proj_zp)
                        assert len(shared_weights_list) in [0, 2]

                        concat_shared_weight = torch.concat(shared_weights_list, 0)
                        concat_shared_scale = torch.concat(shared_scale_list, 0)
                        concat_shared_zp = (
                            torch.concat(shared_zp_list, 0)
                            if len(shared_zp_list) > 0
                            else None
                        )
                        (
                            pack_shared_weight,
                            pack_shared_scale,
                            shared_zp,
                            shared_comp,
                        ) = woq_pack(
                            concat_shared_weight,
                            concat_shared_scale,
                            concat_shared_zp,
                            self.woq_group_size,
                            self.woq_weight_dtype,
                            self.woq_lowp_mode,
                        )
                        w13_shared_qweight_list.append(pack_shared_weight)
                        w13_shared_scale_list.append(pack_shared_scale)
                        if shared_zp is not None:
                            w13_shared_zp_list.append(shared_zp)
                        if is_da8w8 and shared_comp is not None:
                            w13_shared_compensation_list.append(shared_comp)
                        del self.__dict__["_modules"]["mlp"].shared_experts.gate_proj
                        del self.__dict__["_modules"]["mlp"].shared_experts.up_proj

                        shared_moe_down_proj = (
                            module.mlp.shared_experts.down_proj._op_context.get_weight()
                        )
                        shared_moe_down_proj_scale = (
                            module.mlp.shared_experts.down_proj._op_context.get_scales()
                        )
                        shared_moe_down_proj_zp = (
                            module.mlp.shared_experts.down_proj._op_context.get_zero_points()
                        )
                        shared_moe_down_proj_plain = (
                            module.mlp.shared_experts.down_proj._op_context.to_public(
                                shared_moe_down_proj
                            )
                        )
                        w2_qweight, w2_scale, w2_zp, w2_comp = woq_pack(
                            shared_moe_down_proj_plain,
                            shared_moe_down_proj_scale,
                            shared_moe_down_proj_zp,
                            self.woq_group_size,
                            self.woq_weight_dtype,
                            self.woq_lowp_mode,
                        )

                        w2_shared_qweight_list.append(w2_qweight)
                        w2_shared_scale_list.append(w2_scale)
                        if w2_zp is not None:
                            w2_shared_zp_list.append(w2_zp)
                        if is_da8w8 and w2_comp is not None:
                            w2_shared_compensation_list.append(w2_comp)
                        del self.__dict__["_modules"]["mlp"].shared_experts.down_proj

                        self.w13_shared_weight = torch.stack(
                            w13_shared_qweight_list
                        ).detach()
                        self.w13_shared_scale = (
                            torch.stack(w13_shared_scale_list)
                            .detach()
                            .to(torch.float if is_da8w8 else dtype)
                        )
                        self.w13_shared_zp = (
                            torch.stack(w13_shared_zp_list).detach().to(dtype)
                            if len(w13_shared_zp_list) > 0
                            else None
                        )
                        self.w13_shared_compensation = (
                            torch.stack(w13_shared_compensation_list).detach()
                            if len(w13_shared_compensation_list) > 0
                            else None
                        )
                        self.w2_shared_weight = torch.stack(
                            w2_shared_qweight_list
                        ).detach()
                        self.w2_shared_scale = (
                            torch.stack(w2_shared_scale_list)
                            .detach()
                            .to(torch.float if is_da8w8 else dtype)
                        )
                        self.w2_shared_zp = (
                            torch.stack(w2_shared_zp_list).detach().to(dtype)
                            if len(w2_shared_zp_list) > 0
                            else None
                        )
                        self.w2_shared_compensation = (
                            torch.stack(w2_shared_compensation_list).detach()
                            if len(w2_shared_compensation_list) > 0
                            else None
                        )

                        print(
                            "[INFO] Using fused shared MOE WOQ INT8 lowbit weights path..."
                        )
                    else:
                        if (
                            (self.use_fused_moe or self.use_fused_moe_woq)
                            and self.mlp.shared_experts.w13_shared_weight.device.type
                            != "meta"
                        ):
                            dtype = self.mlp.shared_experts.w13_shared_weight.dtype
                            if not self.use_fused_moe_woq:
                                w13_shared_weight = torch.stack(
                                    [self.mlp.shared_experts.w13_shared_weight]
                                ).detach()
                                self.w13_shared_weight = (
                                    torch.ops.torch_ipex.convert_weight_packed_bf16(
                                        w13_shared_weight
                                    )
                                )
                                del self.mlp.shared_experts.w13_shared_weight
                                w2_shared_weight = torch.stack(
                                    [self.mlp.shared_experts.w2_shared_weight]
                                ).detach()
                                self.w2_shared_weight = (
                                    torch.ops.torch_ipex.convert_weight_packed_bf16(
                                        w2_shared_weight
                                    )
                                )
                                del self.mlp.shared_experts.w2_shared_weight
                                # dummy scale/zps
                                self.w13_shared_scale = None
                                self.w13_shared_zp = None
                                self.w13_shared_compensation = None
                                self.w2_shared_scale = None
                                self.w2_shared_zp = None
                                self.w2_shared_compensation = None
                                print("[INFO] Using fused shared MOE bf16 path...")
                            else:
                                w13_shared_qweight_list = []
                                w13_shared_scale_list = []
                                w13_shared_zp_list = []
                                w13_shared_compensation_list = []
                                w2_shared_qweight_list = []
                                w2_shared_scale_list = []
                                w2_shared_zp_list = []
                                w2_shared_compensation_list = []
                                self.woq_weight_dtype = self.qconfig.weight_dtype
                                self.woq_group_size = self.qconfig.group_size
                                self.woq_lowp_mode = self.qconfig.lowp_mode
                                sym_quant_weight = (
                                    self.qconfig.weight_qscheme
                                    == WoqWeightQScheme.SYMMETRIC
                                )
                                is_da8w8 = (
                                    self.woq_weight_dtype == WoqWeightDtype.INT8
                                    and self.woq_lowp_mode == WoqLowpMode.INT8
                                )
                                assert (
                                    self.woq_weight_dtype is WoqWeightDtype.INT8
                                ), "DeepSeek only supports WOQ WoqWeightDtype.INT8..."
                                assert self.woq_lowp_mode in [
                                    WoqLowpMode.INT8,
                                    WoqLowpMode.BF16,
                                ], "DeepSeek only supports WOQ LowpMode in [WoqLowpMode.INT8, WoqLowpMode.BF16]..."
                                (
                                    w13_shared_qweight,
                                    w13_shared_scale,
                                    w13_shared_zp,
                                    w13_shared_comp,
                                ) = woq_quant_and_pack(
                                    self.mlp.shared_experts.w13_shared_weight,
                                    self.woq_group_size,
                                    self.woq_weight_dtype,
                                    self.woq_lowp_mode,
                                    sym_quant_weight,
                                )
                                del self.mlp.shared_experts.w13_shared_weight
                                w13_shared_qweight_list.append(w13_shared_qweight)
                                w13_shared_scale_list.append(w13_shared_scale)
                                if w13_shared_zp is not None:
                                    w13_shared_zp_list.append(w13_shared_zp)
                                if is_da8w8 and w13_shared_comp is not None:
                                    w13_shared_compensation_list.append(w13_shared_comp)

                                (
                                    w2_shared_qweight,
                                    w2_shared_scale,
                                    w2_shared_zp,
                                    w2_shared_comp,
                                ) = woq_quant_and_pack(
                                    self.mlp.shared_experts.w2_shared_weight,
                                    self.woq_group_size,
                                    self.woq_weight_dtype,
                                    self.woq_lowp_mode,
                                    sym_quant_weight,
                                )
                                del self.mlp.shared_experts.w2_shared_weight
                                w2_shared_qweight_list.append(w2_shared_qweight)
                                w2_shared_scale_list.append(w2_shared_scale)
                                if w2_shared_zp is not None:
                                    w2_shared_zp_list.append(w2_shared_zp)
                                if w2_shared_comp is not None:
                                    w2_shared_compensation_list.append(w2_shared_comp)

                                self.w13_shared_weight = torch.stack(
                                    w13_shared_qweight_list
                                ).detach()
                                self.w13_shared_scale = (
                                    torch.stack(w13_shared_scale_list)
                                    .detach()
                                    .to(torch.float if is_da8w8 else dtype)
                                )
                                self.w13_shared_zp = (
                                    torch.stack(w13_shared_zp_list).detach().to(dtype)
                                    if len(w13_shared_zp_list) > 0
                                    else None
                                )
                                self.w13_shared_compensation = (
                                    torch.stack(w13_shared_compensation_list).detach()
                                    if len(w13_shared_compensation_list) > 0
                                    else None
                                )
                                self.w2_shared_weight = torch.stack(
                                    w2_shared_qweight_list
                                ).detach()
                                self.w2_shared_scale = (
                                    torch.stack(w2_shared_scale_list)
                                    .detach()
                                    .to(torch.float if is_da8w8 else dtype)
                                )
                                self.w2_shared_zp = (
                                    torch.stack(w2_shared_zp_list).detach().to(dtype)
                                    if len(w2_shared_zp_list) > 0
                                    else None
                                )
                                self.w2_shared_compensation = (
                                    torch.stack(w2_shared_compensation_list).detach()
                                    if len(w2_shared_compensation_list) > 0
                                    else None
                                )

                                print("[INFO] Using fused shared MOE WOQ INT8 path...")

                if hasattr(self.mlp, "experts"):
                    self.woq_weight_dtype = WoqWeightDtype.INT8  # not used if not WOQ
                    self.woq_group_size = -1  # not used if not WOQ
                    self.woq_lowp_mode = WoqLowpMode.NONE  # not used if not WOQ
                    self.woq_weight_qscheme = (
                        WoqWeightQScheme.UNDEFINED
                    )  # not used if not WOQ
                    if (
                        hasattr(self, "deepseek_lowbit_load")
                        and self.deepseek_lowbit_load
                    ):
                        w13_qweight_list = []
                        w13_scale_list = []
                        w13_zp_list = []
                        w13_compensation_list = []
                        w2_qweight_list = []
                        w2_scale_list = []
                        w2_zp_list = []
                        w2_compensation_list = []
                        dtype = torch.bfloat16

                        self.woq_weight_dtype = module.mlp.experts[0].gate_proj.dtype
                        self.woq_group_size = module.mlp.experts[
                            0
                        ].gate_proj._group_size
                        self.woq_lowp_mode = module.mlp.experts[0].gate_proj._lowp_mode
                        self.woq_weight_qscheme = module.mlp.experts[
                            0
                        ].gate_proj._weight_qscheme
                        is_da8w8 = (
                            self.woq_weight_dtype == WoqWeightDtype.INT8
                            and self.woq_lowp_mode == WoqLowpMode.INT8
                        )
                        assert (
                            self.woq_group_size == -1
                        ), "Block wise WOQ fusedMoE is not supported yet..."
                        for idx in range(config.n_routed_experts):
                            moe_gate_proj_weight = module.mlp.experts[
                                idx
                            ].gate_proj._op_context.to_public(
                                module.mlp.experts[
                                    idx
                                ].gate_proj._op_context.get_weight()
                            )
                            moe_gate_proj_scale = module.mlp.experts[
                                idx
                            ].gate_proj._op_context.get_scales()
                            moe_gate_proj_zp = module.mlp.experts[
                                idx
                            ].gate_proj._op_context.get_zero_points()

                            moe_up_proj_weight = module.mlp.experts[
                                idx
                            ].up_proj._op_context.to_public(
                                module.mlp.experts[idx].up_proj._op_context.get_weight()
                            )
                            moe_up_proj_scale = module.mlp.experts[
                                idx
                            ].up_proj._op_context.get_scales()
                            moe_up_proj_zp = module.mlp.experts[
                                idx
                            ].up_proj._op_context.get_zero_points()

                            weights_list = [
                                moe_gate_proj_weight,
                                moe_up_proj_weight,
                            ]
                            scale_list = [
                                moe_gate_proj_scale,
                                moe_up_proj_scale,
                            ]
                            zp_list = [
                                moe_gate_proj_zp,
                                moe_up_proj_zp,
                            ]

                            concat_weight = torch.concat(weights_list, 0)
                            concat_scale = torch.concat(scale_list, 0)
                            concat_zp = (
                                torch.concat(zp_list, 0)
                                if zp_list[0] is not None
                                else None
                            )
                            del self.__dict__["_modules"]["mlp"].experts[idx].gate_proj
                            del self.__dict__["_modules"]["mlp"].experts[idx].up_proj
                            w13_qweight, w13_scale, w13_zp, w13_comp = woq_pack(
                                concat_weight,
                                concat_scale,
                                concat_zp,
                                self.woq_group_size,
                                self.woq_weight_dtype,
                                self.woq_lowp_mode,
                            )
                            w13_qweight_list.append(w13_qweight)
                            w13_scale_list.append(w13_scale)
                            if w13_zp is not None:
                                w13_zp_list.append(w13_zp)
                            if is_da8w8 and w13_comp is not None:
                                w13_compensation_list.append(w13_comp)

                            moe_down_proj_weight = module.mlp.experts[
                                idx
                            ].down_proj._op_context.get_weight()
                            moe_down_proj_scale = module.mlp.experts[
                                idx
                            ].down_proj._op_context.get_scales()
                            moe_down_proj_zp = module.mlp.experts[
                                idx
                            ].down_proj._op_context.get_zero_points()
                            moe_down_proj_plain_weight = module.mlp.experts[
                                idx
                            ].down_proj._op_context.to_public(moe_down_proj_weight)
                            w2_qweight, w2_scale, w2_zp, w2_comp = woq_pack(
                                moe_down_proj_plain_weight,
                                moe_down_proj_scale,
                                moe_down_proj_zp,
                                self.woq_group_size,
                                self.woq_weight_dtype,
                                self.woq_lowp_mode,
                            )
                            w2_qweight_list.append(moe_down_proj_weight)
                            w2_scale_list.append(moe_down_proj_scale)
                            if w2_zp is not None:
                                w2_zp_list.append(w2_zp)
                            if is_da8w8 and w2_comp is not None:
                                w2_compensation_list.append(w2_comp)
                            del self.__dict__["_modules"]["mlp"].experts[idx].down_proj
                        if self.unify_experts:
                            w13_qweight_list.append(self.w13_shared_weight[0])
                            del self.w13_shared_weight
                            w13_scale_list.append(self.w13_shared_scale[0])
                            del self.w13_shared_scale
                            if self.w13_shared_zp is not None:
                                w13_zp_list.append(self.w13_shared_zp[0])
                                del self.w13_shared_zp
                            if self.w13_shared_compensation is not None:
                                w13_compensation_list.append(
                                    self.w13_shared_compensation[0]
                                )
                                del self.w13_shared_compensation
                            w2_qweight_list.append(self.w2_shared_weight[0])
                            del self.w2_shared_weight
                            w2_scale_list.append(self.w2_shared_scale[0])
                            del self.w2_shared_scale
                            if self.w2_shared_zp is not None:
                                w2_zp_list.append(self.w2_shared_zp[0])
                                del self.w2_shared_zp
                            if self.w2_shared_compensation is not None:
                                w2_compensation_list.append(
                                    self.w2_shared_compensation[0]
                                )
                                del self.w2_shared_compensation
                        self.w13_weight = torch.stack(w13_qweight_list).detach()
                        self.w13_scale = (
                            torch.stack(w13_scale_list)
                            .detach()
                            .to(torch.float if is_da8w8 else dtype)
                        )
                        self.w13_zp = (
                            torch.stack(w13_zp_list).detach().to(dtype)
                            if len(w13_zp_list) > 0
                            else None
                        )
                        self.w13_compensation = (
                            torch.stack(w13_compensation_list).detach()
                            if len(w13_compensation_list) > 0
                            else None
                        )
                        self.w2_weight = torch.stack(w2_qweight_list).detach()
                        self.w2_scale = (
                            torch.stack(w2_scale_list)
                            .detach()
                            .to(torch.float if is_da8w8 else dtype)
                        )
                        self.w2_zp = (
                            torch.stack(w2_zp_list).detach().to(dtype)
                            if len(w2_zp_list) > 0
                            else None
                        )
                        self.w2_compensation = (
                            torch.stack(w2_compensation_list).detach()
                            if len(w2_compensation_list) > 0
                            else None
                        )

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
                                    torch.ops.torch_ipex.convert_weight_packed_bf16(
                                        w13_weight
                                    )
                                )
                                for idx in range(len(self.mlp.experts)):
                                    del self.mlp.experts[idx].w13_weight
                                w2_weight = torch.stack(
                                    [
                                        self.mlp.experts[idx].w2_weight
                                        for idx in range(len(self.mlp.experts))
                                    ]
                                ).detach()
                                self.w2_weight = (
                                    torch.ops.torch_ipex.convert_weight_packed_bf16(
                                        w2_weight
                                    )
                                )
                                for idx in range(len(self.mlp.experts)):
                                    del self.mlp.experts[idx].w2_weight
                                # dummy scale/zps

                                self.w13_scale = None
                                self.w13_zp = None
                                self.w13_compensation = None
                                self.w2_scale = None
                                self.w2_zp = None
                                self.w2_compensation = None
                                print("[INFO] Using fused MOE bf16 path...")
                            else:
                                w13_qweight_list = []
                                w13_scale_list = []
                                w13_zp_list = []
                                w13_compensation_list = []
                                w2_qweight_list = []
                                w2_scale_list = []
                                w2_zp_list = []
                                w2_compensation_list = []
                                self.woq_weight_dtype = self.qconfig.weight_dtype
                                self.woq_group_size = self.qconfig.group_size
                                self.woq_lowp_mode = self.qconfig.lowp_mode
                                sym_quant_weight = (
                                    self.qconfig.weight_qscheme
                                    == WoqWeightQScheme.SYMMETRIC
                                )
                                is_da8w8 = (
                                    self.woq_weight_dtype == WoqWeightDtype.INT8
                                    and self.woq_lowp_mode == WoqLowpMode.INT8
                                )
                                assert (
                                    self.woq_weight_dtype is WoqWeightDtype.INT8
                                ), "DeepSeek only supports WOQ WoqWeightDtype.INT8..."
                                assert self.woq_lowp_mode in [
                                    WoqLowpMode.INT8,
                                    WoqLowpMode.BF16,
                                ], "DeepSeek only supports WOQ LowpMode in [WoqLowpMode.INT8, WoqLowpMode.BF16]..."
                                for idx in range(len(self.mlp.experts)):
                                    w13_qweight, w13_scale, w13_zp, w13_comp = (
                                        woq_quant_and_pack(
                                            self.mlp.experts[idx].w13_weight,
                                            self.woq_group_size,
                                            self.woq_weight_dtype,
                                            self.woq_lowp_mode,
                                            sym_quant_weight,
                                        )
                                    )
                                    del self.mlp.experts[idx].w13_weight
                                    w13_qweight_list.append(w13_qweight)
                                    w13_scale_list.append(w13_scale)
                                    if w13_zp is not None:
                                        w13_zp_list.append(w13_zp)
                                    if is_da8w8 and w13_comp is not None:
                                        w13_compensation_list.append(w13_comp)

                                    w2_qweight, w2_scale, w2_zp, w2_comp = (
                                        woq_quant_and_pack(
                                            self.mlp.experts[idx].w2_weight,
                                            self.woq_group_size,
                                            self.woq_weight_dtype,
                                            self.woq_lowp_mode,
                                            sym_quant_weight,
                                        )
                                    )
                                    del self.mlp.experts[idx].w2_weight
                                    w2_qweight_list.append(w2_qweight)
                                    w2_scale_list.append(w2_scale)
                                    if w2_zp is not None:
                                        w2_zp_list.append(w2_zp)
                                    if is_da8w8 and w2_comp is not None:
                                        w2_compensation_list.append(w2_comp)

                                if self.unify_experts:
                                    w13_qweight_list.append(self.w13_shared_weight[0])
                                    del self.w13_shared_weight
                                    w13_scale_list.append(self.w13_shared_scale[0])
                                    del self.w13_shared_scale
                                    if self.w13_shared_zp is not None:
                                        w13_zp_list.append(self.w13_shared_zp[0])
                                        del self.w13_shared_zp
                                    if self.w13_shared_compensation is not None:
                                        w13_compensation_list.append(
                                            self.w13_shared_compensation[0]
                                        )
                                        del self.w13_shared_compensation
                                    w2_qweight_list.append(self.w2_shared_weight[0])
                                    del self.w2_shared_weight
                                    w2_scale_list.append(self.w2_shared_scale[0])
                                    del self.w2_shared_scale
                                    if self.w2_shared_zp is not None:
                                        w2_zp_list.append(self.w2_shared_zp[0])
                                        del self.w2_shared_zp
                                    if self.w2_shared_compensation is not None:
                                        w2_compensation_list.append(
                                            self.w2_shared_compensation[0]
                                        )
                                        del self.w2_shared_compensation
                                self.w13_weight = torch.stack(w13_qweight_list).detach()
                                self.w13_scale = (
                                    torch.stack(w13_scale_list)
                                    .detach()
                                    .to(torch.float if is_da8w8 else dtype)
                                )
                                self.w13_zp = (
                                    torch.stack(w13_zp_list).detach().to(dtype)
                                    if len(w13_zp_list) > 0
                                    else None
                                )
                                self.w13_compensation = (
                                    torch.stack(w13_compensation_list).detach()
                                    if len(w13_compensation_list) > 0
                                    else None
                                )
                                self.w2_weight = torch.stack(w2_qweight_list).detach()
                                self.w2_scale = (
                                    torch.stack(w2_scale_list)
                                    .detach()
                                    .to(torch.float if is_da8w8 else dtype)
                                )
                                self.w2_zp = (
                                    torch.stack(w2_zp_list).detach().to(dtype)
                                    if len(w2_zp_list) > 0
                                    else None
                                )
                                self.w2_compensation = (
                                    torch.stack(w2_compensation_list).detach()
                                    if len(w2_compensation_list) > 0
                                    else None
                                )

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
                            offset = (
                                0
                                if self.model_backbone == "Qwen3MoeForCausalLM"
                                else self.mlp.ep_rank * self.mlp.experts_per_rank
                            )
                            for expert_idx in range(len(self.mlp.experts)):
                                expert_layer = self.mlp.experts[expert_idx + offset]
                                if self.moe_linear_type in [0, 1]:
                                    self.gate_weights.append(
                                        expert_layer.gate_proj.weight
                                    )
                                    self.up_weights.append(expert_layer.up_proj.weight)
                                    self.down_weights.append(
                                        expert_layer.down_proj.weight
                                    )
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
                                    self.gate_ctx.append(
                                        expert_layer.gate_proj._op_context
                                    )
                                    self.up_ctx.append(expert_layer.up_proj._op_context)
                                    self.down_ctx.append(
                                        expert_layer.down_proj._op_context
                                    )
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
