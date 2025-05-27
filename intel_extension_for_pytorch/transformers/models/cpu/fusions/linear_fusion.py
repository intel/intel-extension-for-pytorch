import torch
from torch import nn
from torch.nn.utils import skip_init
from typing import Optional, Callable
import math
from .....utils._logger import logger, WarningType
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.nn.utils._weight_prepack import _IPEXLinear
from intel_extension_for_pytorch.quantization import (
    get_weight_only_quant_qconfig_mapping,
    dequantize_per_channel,
    dequantize_per_block,
)
from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
    _enable_tpp,
    _disable_tpp,
)


class _IPEXlinearFusionCPU(nn.Module):
    def __init__(self, linear, tpp=False, woq=False):
        super().__init__()
        self.tpp = tpp
        self.woq = woq
        self.dtype = None if woq else linear.weight.dtype

    def extra_repr(self):
        extra_repr_str = f"dtype = {self.dtype}, tpp = {self.tpp}, woq = {self.woq}"
        return extra_repr_str


class _IPEXlinearSiluCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.linear = module

    def forward(self, x):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            w = torch.ops.torch_ipex.choose_tpp_linear_weight(
                x, self.linear.weight, self.linear.weight_for_large_batch
            )
            return torch.ops.torch_ipex.tpp_linear_silu(
                x,
                w.detach(),
                (
                    self.linear.bias.detach()
                    if self.linear.bias is not None
                    else x.new_empty(0)
                ),
                self.linear.out_features,
            )
        elif (
            self.woq
            and hasattr(self.linear, "_op_context")
            and self.linear._op_context is not None
        ):
            return torch.ops.torch_ipex.woq_linear_silu(
                x,
                self.linear._op_context.get_data_handle(),
            )
        else:  # fallback path
            return nn.functional.silu(self.linear(x))


class _IPEXlinearReluCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.linear = module

    def forward(self, x):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            w = torch.ops.torch_ipex.choose_tpp_linear_weight(
                x, self.linear.weight, self.linear.weight_for_large_batch
            )
            return torch.ops.torch_ipex.tpp_linear_relu(
                x,
                w.detach(),
                (
                    self.linear.bias.detach()
                    if self.linear.bias is not None
                    else x.new_empty(0)
                ),
                self.linear.out_features,
            )
        elif (
            self.woq
            and hasattr(self.linear, "_op_context")
            and self.linear._op_context is not None
        ):
            return torch.ops.torch_ipex.woq_linear_relu(
                x,
                self.linear._op_context.get_data_handle(),
            )
        else:  # fallback path
            return nn.functional.relu(self.linear(x))


class _IPEXlinearMulCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.linear = module

    def forward(self, x, y):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            y = y.to(self.dtype).contiguous()
            w = torch.ops.torch_ipex.choose_tpp_linear_weight(
                x, self.linear.weight, self.linear.weight_for_large_batch
            )
            return torch.ops.torch_ipex.tpp_linear_mul(
                x,
                y,
                w.detach(),
                (
                    self.linear.bias.detach()
                    if self.linear.bias is not None
                    else x.new_empty(0)
                ),
                self.linear.out_features,
            )
        elif (
            self.woq
            and hasattr(self.linear, "_op_context")
            and self.linear._op_context is not None
        ):
            return torch.ops.torch_ipex.woq_linear_mul(
                x,
                self.linear._op_context.get_data_handle(),
                [y],
            )
        else:  # fallback path
            return self.linear(x) * y


class _IPEXlinearAddCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.linear = module

    def forward(self, x, y):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            y = y.to(self.dtype).contiguous()
            w = torch.ops.torch_ipex.choose_tpp_linear_weight(
                x, self.linear.weight, self.linear.weight_for_large_batch
            )
            return torch.ops.torch_ipex.tpp_linear_add(
                x,
                y,
                w.detach(),
                (
                    self.linear.bias.detach()
                    if self.linear.bias is not None
                    else x.new_empty(0)
                ),
                1.0,
                self.linear.out_features,
            )
        if (
            self.woq
            and hasattr(self.linear, "_op_context")
            and self.linear._op_context is not None
        ):
            return torch.ops.torch_ipex.woq_linear_add(
                x,
                self.linear._op_context.get_data_handle(),
                [y],
            )
        else:  # fallback path
            return self.linear(x) + y


class _IPEXlinearAddAddCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.linear = module

    def forward(self, x, y, z):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            y = y.to(self.dtype).contiguous()
            z = z.to(self.dtype).contiguous()
            w = torch.ops.torch_ipex.choose_tpp_linear_weight(
                x, self.linear.weight, self.linear.weight_for_large_batch
            )
            return torch.ops.torch_ipex.tpp_linear_add_add(
                x,
                y,
                z,
                w.detach(),
                (
                    self.linear.bias.detach()
                    if self.linear.bias is not None
                    else x.new_empty(0)
                ),
                1.0,
                self.linear.out_features,
            )
        if (
            self.woq
            and hasattr(self.linear, "_op_context")
            and self.linear._op_context is not None
        ):
            return torch.ops.torch_ipex.woq_linear_add_add(
                x,
                self.linear._op_context.get_data_handle(),
                [y, z],
            )
        else:  # fallback path
            return self.linear(x) + y + z


class _IPEXlinearNewGeluCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.linear = module

    def forward(self, x):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            w = torch.ops.torch_ipex.choose_tpp_linear_weight(
                x, self.linear.weight, self.linear.weight_for_large_batch
            )
            return torch.ops.torch_ipex.tpp_linear_gelu(
                x,
                w.detach(),
                (
                    self.linear.bias.detach()
                    if self.linear.bias is not None
                    else x.new_empty(0)
                ),
                self.linear.out_features,
            )
        elif (
            self.woq
            and hasattr(self.linear, "_op_context")
            and self.linear._op_context is not None
        ):
            return torch.ops.torch_ipex.woq_linear_new_gelu(
                x,
                self.linear._op_context.get_data_handle(),
            )
        else:  # fallback path
            x = self.linear(x)
            return (
                0.5
                * x
                * (
                    1.0
                    + torch.tanh(
                        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                    )
                )
            )


class _IPEXlinearGeluCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.linear = module
        self.gelu = nn.GELU()

    def forward(self, x):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            w = torch.ops.torch_ipex.choose_tpp_linear_weight(
                x, self.linear.weight, self.linear.weight_for_large_batch
            )
            return torch.ops.torch_ipex.tpp_linear_gelu(
                x,
                w.detach(),
                (
                    self.linear.bias.detach()
                    if self.linear.bias is not None
                    else x.new_empty(0)
                ),
                self.linear.out_features,
            )
        if (
            self.woq
            and hasattr(self.linear, "_op_context")
            and self.linear._op_context is not None
        ):
            return torch.ops.torch_ipex.woq_linear_gelu(
                x,
                self.linear._op_context.get_data_handle(),
            )
        else:  # fallback path
            x = self.gelu(self.linear(x))
            return x


class _IPEXConcatLinearCPU(_IPEXlinearFusionCPU):
    def __init__(self, module, tpp=False, woq=False):
        assert hasattr(module, "linear_0")
        super().__init__(module.linear_0, tpp=tpp, woq=woq)
        assert hasattr(module, "num_concat")
        self.num_concat = module.num_concat
        self.concat_linear = None
        self.linear_list = []
        self.woq = woq
        self.tpp = tpp
        use_g_idx = False
        from intel_extension_for_pytorch.nn.modules import WeightOnlyQuantizedLinear

        if woq:
            for i in range(self.num_concat):
                attr_name = f"linear_{i}"
                assert hasattr(module, attr_name)
                self.linear_list.append(getattr(module, attr_name))
                gidx = self.linear_list[-1]._op_context.get_g_idx()
                use_g_idx = use_g_idx or (gidx is not None)
        if (
            woq
            and (not use_g_idx)
            and all(
                isinstance(linear, WeightOnlyQuantizedLinear)
                for linear in self.linear_list
            )
        ):
            # Quantization is done before lowering to CPU.
            # We assume weights are all in shape [N, K].
            # We need to unpack weights then concat them
            weights_list = []
            scales_list = []
            zeros_list = []
            bias_list = []
            w_dtype = self.linear_list[0].dtype
            lowp_mode = self.linear_list[0]._lowp_mode
            act_quant_mode = self.linear_list[0]._act_quant_mode
            group_size = self.linear_list[0]._group_size
            cache_weight_for_large_batch = self.linear_list[
                0
            ]._cache_weight_for_large_batch
            weight_qscheme = self.linear_list[0]._weight_qscheme
            qconfig_mapping = get_weight_only_quant_qconfig_mapping(
                weight_dtype=w_dtype,
                lowp_mode=lowp_mode,
                act_quant_mode=act_quant_mode,
                group_size=group_size,
                weight_qscheme=weight_qscheme,
            )
            if cache_weight_for_large_batch:
                from intel_extension_for_pytorch.utils.weight_only_quantization import (
                    _woq_enable_weight_cache_for_large_batch,
                )

                qconfig_mapping = _woq_enable_weight_cache_for_large_batch(
                    qconfig_mapping
                )
            qconfig = qconfig_mapping.global_qconfig
            for i in range(self.num_concat):
                linear = self.linear_list[i]
                if not hasattr(linear, "_op_context"):
                    logger.warning(
                        "Concat linear fusion for CPU WOQ failed "
                        + "because linear is not converted to WOQ Linear. "
                        + "Falling back to separate linears.",
                        _type=WarningType.NotSupported,
                    )
                    weights_list = []
                    break
                qw = linear._op_context.to_public(linear._op_context.get_weight())
                scales = linear._op_context.get_scales()
                zero_points = linear._op_context.get_zero_points()
                weight_shape = linear._op_context.get_weight_shape()
                if group_size > 0:
                    weights_list.append(
                        dequantize_per_block(
                            qw, scales, zero_points, w_dtype, group_size, weight_shape
                        )
                    )
                else:
                    weights_list.append(
                        dequantize_per_channel(
                            qw, scales, zero_points, w_dtype, weight_shape
                        )
                    )
                # OC of Weight may be padded to a multiple of block_n. So are scales and zero points.
                bias = linear._op_context.get_bias()
                assert zero_points is None or scales.shape == zero_points.shape
                assert bias is None or bias.shape[0] == scales.shape[0]
                if weight_shape[0] < scales.shape[0]:
                    original_n = weight_shape[0]
                    scales_list.append(scales.narrow(0, 0, original_n).contiguous())
                    if zero_points is not None:
                        zeros_list.append(
                            zero_points.narrow(0, 0, original_n).contiguous()
                        )
                    bias_list.append(bias.narrow(0, 0, original_n).contiguous())
                else:
                    assert weight_shape[0] == scales.shape[0]
                    scales_list.append(scales)
                    if zero_points is not None:
                        zeros_list.append(zero_points)
                    bias_list.append(bias)
                w_dtype = linear.dtype
            if weights_list:
                concat_weight = torch.concat(weights_list, 0)
                concat_scales = torch.concat(scales_list, 0)
                if len(zeros_list) > 0:
                    concat_zeros = torch.concat(zeros_list, 0)
                else:
                    concat_zeros = None
                use_bias = all((b is not None for b in bias_list))
                concat_bias = torch.concat(bias_list, 0) if use_bias else None
                mod = nn.Linear(
                    concat_weight.shape[1], concat_weight.shape[0], use_bias
                )
                mod.weight = nn.Parameter(concat_weight)
                mod.bias = nn.Parameter(concat_bias) if use_bias else None
                mod.qconfig = qconfig
                self.concat_linear = WeightOnlyQuantizedLinear.from_float(
                    mod, concat_scales, concat_zeros
                )
        elif hasattr(module, "concat_linear") and module.concat_linear is not None:
            self.concat_linear = module.concat_linear
        else:
            for i in range(self.num_concat):
                attr_name = f"linear_{i}"
                setattr(self, attr_name, getattr(module, attr_name))

    def forward(self, x):
        if self.concat_linear is not None:
            return self.concat_linear(x)

        output_list = []
        for i in range(self.num_concat):
            assert hasattr(self, f"linear_{i}")
            linear = getattr(self, f"linear_{i}")
            y = linear(x)
            output_list.append(y)
        return tuple(output_list)


class _IPEXlinearSiluMulCPU(nn.Module):
    def __init__(self, module_s, module_m, tpp=False, woq=False):
        super().__init__()
        self.tpp = tpp
        self.woq = woq
        self.linear_s = module_s
        self.linear_m = module_m
        self.dtype = module_s.weight.dtype if self.tpp else None

    def forward(self, x):
        if (
            self.tpp
            and not self.linear_s.tpp_fallback
            and not self.linear_m.tpp_fallback
        ):
            x = x.to(self.dtype).contiguous()
            w_s = torch.ops.torch_ipex.choose_tpp_linear_weight(
                x, self.linear_s.weight, self.linear_s.weight_for_large_batch
            )
            w_m = torch.ops.torch_ipex.choose_tpp_linear_weight(
                x, self.linear_m.weight, self.linear_m.weight_for_large_batch
            )
            return torch.ops.torch_ipex.tpp_fused_gate_up_proj(
                x,
                w_s.detach(),
                (
                    self.linear_s.bias.detach()
                    if self.linear_s.bias is not None
                    else x.new_empty(0)
                ),
                w_m.detach(),
                (
                    self.linear_m.bias.detach()
                    if self.linear_m.bias is not None
                    else x.new_empty(0)
                ),
            )
        elif (
            self.woq
            and hasattr(self.linear_s, "_op_context")
            and self.linear_s._op_context is not None
            and hasattr(self.linear_m, "_op_context")
            and self.linear_m._op_context is not None
        ):
            y = torch.ops.torch_ipex.woq_linear_silu(
                x,
                self.linear_s._op_context.get_data_handle(),
            )
            return torch.ops.torch_ipex.woq_linear_mul(
                x,
                self.linear_m._op_context.get_data_handle(),
                [y],
            )
        else:  # fallback path
            return nn.functional.silu(self.linear_s(x)) * self.linear_m(x)


class _IPEXlinearSiluAndMulCPU(nn.Module):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__()
        self.tpp = tpp
        self.woq = woq
        self.linear = module
        self.dtype = module.weight.dtype if self.tpp else None

    def forward(self, x, y):
        if self.tpp and not self.linear.tpp_fallback:
            x = x.to(self.dtype).contiguous()
            w = torch.ops.torch_ipex.choose_tpp_linear_weight(
                x, self.linear.weight, self.linear.weight_for_large_batch
            )
            x1 = torch.ops.torch_ipex.tpp_linear_silu(
                x,
                w.detach(),
                (
                    self.linear.bias.detach()
                    if self.linear.bias is not None
                    else x.new_empty(0)
                ),
                self.linear.out_features,
            )
            return x1 * y
        elif (
            self.woq
            and hasattr(self.linear, "_op_context")
            and self.linear._op_context is not None
        ):
            return (
                torch.ops.torch_ipex.woq_linear_silu(
                    x,
                    self.linear._op_context.get_data_handle(),
                )
                * y
            )
        else:  # fallback path
            return nn.functional.silu(self.linear(x)) * y


class _IPEXGatedMLPMOECPU(nn.Module):
    def __init__(
        self,
        W13,
        W2,
        W3=None,
        use_prepack=False,
        w1_scale_inv=None,
        w2_scale_inv=None,
        a1_scale_inv=None,
        a2_scale_inv=None,
    ):
        super().__init__()

        self.num_experts = W2.shape[0]
        self.hidden_size = W2.shape[1]
        self.intermediate_size = W2.shape[2]
        linear_list = []
        for i in range(W2.shape[0]):
            if W3 is not None:
                _W1 = W13[i]
                _W3 = W3[i]
            else:
                _W1 = W13[i][0 : self.intermediate_size, :]
                _W3 = W13[i][self.intermediate_size : 2 * self.intermediate_size, :]
            linear1 = skip_init(
                torch.nn.Linear,
                self.intermediate_size,
                self.hidden_size,
                bias=False,
            )
            linear1.weight = nn.Parameter(_W1)
            linear2 = skip_init(
                torch.nn.Linear,
                self.hidden_size,
                self.intermediate_size,
                bias=False,
            )
            linear2.weight = nn.Parameter(W2[i])
            linear3 = skip_init(
                torch.nn.Linear,
                self.intermediate_size,
                self.hidden_size,
                bias=False,
            )
            linear3.weight = nn.Parameter(_W3)
            linear_per_expert = nn.ModuleList([linear1, linear2, linear3])
            linear_list.append(linear_per_expert)
        self.linear_module_list = nn.ModuleList(
            [linear_list[i] for i in range(W2.shape[0])]
        )
        if use_prepack:
            _disable_tpp()
            if W13.dtype is torch.bfloat16 and W2.dtype is torch.bfloat16:
                _enable_tpp()
                auto_kernel_selection = False
            else:
                auto_kernel_selection = True
            self.linear_module_list = ipex.optimize(
                self.linear_module_list.eval(),
                dtype=W13.dtype,
                auto_kernel_selection=auto_kernel_selection,
                inplace=True,
            )

    # This is used by the Deepseek-V2 and Deepseek-V3 model
    # ref from:
    # https://github.com/simon-mo/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L480
    def grouped_topk(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        num_expert_group: int = 0,
        topk_group: int = 0,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ):

        assert (
            hidden_states.shape[0] == gating_output.shape[0]
        ), "Number of tokens mismatch"

        if scoring_func == "softmax":
            scores = torch.softmax(gating_output, dim=-1)
        elif scoring_func == "sigmoid":
            scores = gating_output.sigmoid()
        else:
            raise ValueError(f"Unsupported scoring function: {scoring_func}")

        if e_score_correction_bias is not None:
            scores.add_(e_score_correction_bias.unsqueeze(0))

        num_token = scores.shape[0]
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )  # [n, n_group]
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
            .reshape(num_token, -1)
        )  # [n, e]
        tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, head_dim = hidden_states.shape
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            routing_weights, selected_experts = self.grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
            )
        elif custom_routing_function is None:
            assert scoring_func == "softmax"
            routing_weights = torch.nn.functional.softmax(
                router_logits, dim=1, dtype=torch.float32
            )
            routing_weights, selected_experts = torch.topk(
                routing_weights, top_k, dim=-1
            )
            if renormalize:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        else:
            routing_weights, selected_experts = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
            )

        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = torch.zeros(
            (batch_size, head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_mask = torch.nn.functional.one_hot(
            selected_experts.to(torch.int64), num_classes=self.num_experts
        ).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if isinstance(self.linear_module_list[expert_idx][0], _IPEXLinear):
                if (
                    hasattr(self.linear_module_list[expert_idx][0], "use_dnnl")
                    and self.linear_module_list[expert_idx][0].use_dnnl
                ):
                    final_hidden_states = torch.ops.torch_ipex.mixtral_moe(
                        hidden_states,
                        top_x,
                        idx,
                        self.linear_module_list[expert_idx][0]._get_forward_weight(),
                        self.linear_module_list[expert_idx][0].ctx.get_data_handle(),
                        self.linear_module_list[expert_idx][2]._get_forward_weight(),
                        self.linear_module_list[expert_idx][2].ctx.get_data_handle(),
                        self.linear_module_list[expert_idx][1]._get_forward_weight(),
                        self.linear_module_list[expert_idx][1].ctx.get_data_handle(),
                        hasattr(self.linear_module_list[expert_idx][0], "use_dnnl")
                        and self.linear_module_list[expert_idx][0].use_dnnl,
                        routing_weights,
                        final_hidden_states,
                        False,
                    )
                else:
                    final_hidden_states = torch.ops.torch_ipex.mixtral_moe_tpp(
                        hidden_states,
                        top_x,
                        idx,
                        self.linear_module_list[expert_idx][0].weight.detach(),
                        self.linear_module_list[expert_idx][2].weight.detach(),
                        self.linear_module_list[expert_idx][1].weight.detach(),
                        (
                            self.linear_module_list[expert_idx][0].tpp_fallback
                            if hasattr(
                                self.linear_module_list[expert_idx][0], "tpp_fallback"
                            )
                            else True
                        ),
                        routing_weights,
                        final_hidden_states,
                        False,
                    )
            else:
                # nn.Linear
                final_hidden_states = torch.ops.torch_ipex.mixtral_moe_tpp(
                    hidden_states,
                    top_x,
                    idx,
                    self.linear_module_list[expert_idx][0].weight.detach(),
                    self.linear_module_list[expert_idx][2].weight.detach(),
                    self.linear_module_list[expert_idx][1].weight.detach(),
                    True,
                    routing_weights,
                    final_hidden_states,
                    False,
                )

        return final_hidden_states.view(-1, head_dim)
