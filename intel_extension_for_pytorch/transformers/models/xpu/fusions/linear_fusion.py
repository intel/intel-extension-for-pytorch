import torch
from torch import nn
from typing import Optional, Callable
import numpy as np
import os


class _IPEXlinearFusionXPU(nn.Module):
    def __init__(self, linear, tpp=False, woq=False):
        super().__init__()
        # Do not support tpp & woq for now
        self.tpp = tpp
        self.woq = woq
        self.dtype = None if woq else linear.weight.dtype

    def extra_repr(self):
        extra_repr_str = f"dtype = {self.dtype}, tpp = {self.tpp}, woq = {self.woq}"
        return extra_repr_str


class _IPEXlinearAddXPU(_IPEXlinearFusionXPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.weight = module.weight.transpose(0, 1).contiguous()
        self.bias = module.bias

    def forward(self, x, y):
        if self.bias is not None:
            x = torch.ops.torch_ipex.mm_bias_resadd(
                x, self.weight, self.bias, 1.0, y, 1.0
            )
        else:
            x = torch.addmm(
                y.flatten(0, -2),
                x.flatten(0, -2),
                self.weight,
                beta=1.0,
            )
        return x


class _IPEXlinearAddAddXPU(_IPEXlinearFusionXPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.weight = module.weight.transpose(0, 1).contiguous()
        self.bias = module.bias

    def forward(self, x, y, z):
        if self.bias is not None:
            x = torch.ops.torch_ipex.mm_bias_resadd(
                x, self.weight, self.bias, 1.0, y, 1.0
            )
            x += z
        else:
            x = torch.ops.torch_ipex.mm_bias_resadd(x, self.weight, z, 1.0, y, 1.0)
        return x


class _IPEXlinearGeluXPU(_IPEXlinearFusionXPU):
    def __init__(self, module, tpp=False, woq=False):
        super().__init__(module, tpp=tpp, woq=woq)
        self.weight = module.weight.transpose(0, 1).contiguous()
        self.bias = module.bias

    def forward(self, x):
        return torch.ops.torch_ipex.matmul_gelu(x, self.weight, self.bias, 1.0, "tanh")


class _IPEXlinearSiluMulXPU(_IPEXlinearFusionXPU):
    def __init__(self, module_1, module_2, tpp=False, woq=False):
        super().__init__(module_1, tpp=tpp, woq=woq)
        self.weight_1 = module_1.weight.transpose(0, 1).contiguous()
        self.weight_2 = module_2.weight.transpose(0, 1).contiguous()
        self.bias_1 = module_1.bias
        self.bias_2 = module_2.bias

    def forward(self, x):
        if self.bias_1 is not None:
            silu = nn.SiLU()
            linear_1_bias = torch.ops.torch_ipex.mm_resadd(
                x, self.weight_1, self.bias_1.unsqueeze(0), 1.0
            )
            silu_mm = silu(linear_1_bias)
        else:
            silu_mm = torch.ops.torch_ipex.matmul_silu(x, self.weight_1)

        if self.bias_2 is not None:
            linear_2_bias = torch.ops.torch_ipex.mm_resadd(
                x, self.weight_2, self.bias_2.unsqueeze(0), 1.0
            )
            return silu_mm * linear_2_bias
        else:
            return torch.ops.torch_ipex.mm_resmul(x, self.weight_2, silu_mm)


class _IPEXGatedMLPMOEXPU(nn.Module):
    # W2: [num_experts, hidden_size, intermediate_size]
    # W13: [num_experts, intermediate_size * 2, hidden_size]
    # W1/W3: [num_experts, intermediate_size, hidden_size]
    # w1_scale_inv: [num_experts]
    # w2_scale_inv: [num_experts]
    # a1_scale_inv: [num_experts]
    # a2_scale_inv: [num_experts]
    # TODO: Only support w1_scale_inv & w2_scale_inv right now
    # XPU does not support cat fp8, so if use fp8, W3 should be None
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
        w13_bias=None,
        w2_bias=None,
        is_mxfp4=False,
        is_fp8=False,
        is_int4=False,
        experts_start_id=0,
    ):
        super().__init__()
        self.num_experts = W2.shape[0]
        if W3 is None:
            self.W13 = torch.transpose(W13, 1, 2)
        else:
            self.W13 = torch.transpose(torch.cat((W13, W3), dim=1), 1, 2)
        self.W2 = torch.transpose(W2, 1, 2)

        self.W2 = self.W2.contiguous()
        self.W13 = self.W13.contiguous()

        self.w13_weight_scale_inv = w1_scale_inv if w1_scale_inv is not None else None
        self.w2_weight_scale_inv = w2_scale_inv if w2_scale_inv is not None else None
        self.w13_input_scale_inv = a1_scale_inv if a1_scale_inv is not None else None
        self.w2_input_scale_inv = a2_scale_inv if a2_scale_inv is not None else None

        self.w13_bias = w13_bias
        self.w2_bias = w2_bias

        self.is_mxfp4 = is_mxfp4
        self.is_fp8 = is_fp8
        self.is_int4 = is_int4
        self.experts_start_id = experts_start_id

        self.moe_gemm_using_native_impl = (
            os.environ.get("IPEX_MOE_GEMM_NATIVE", "0") == "1"
        )

        if self.is_mxfp4 or self.is_int4:
            self.W13 = self.marlin_shuffle_weight(self.W13)
            self.W2 = self.marlin_shuffle_weight(self.W2)
            self.w13_weight_scale_inv.data = torch.transpose(
                w1_scale_inv, 1, 2
            ).contiguous()
            self.w2_weight_scale_inv.data = torch.transpose(
                w2_scale_inv, 1, 2
            ).contiguous()

        # delete original weight to avoid memory pressure
        W13.data = self.W13
        W2.data = self.W2
        if W3 is not None:
            W3.data = self.W13

        torch.xpu.empty_cache()

    def marlin_shuffle_weight(self, qweight):
        # qweight: [E, K, N]
        E, K, N = qweight.shape
        k = K * 8
        shuffled_idx = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        pack_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        shuffled_weight = torch.zeros(
            [E, k // 8, N], dtype=torch.int32, device=qweight.device
        )

        for e in range(E):
            data = qweight[e][[i // 8 for i in range(k)], :]
            shift = (
                torch.tensor(
                    shuffled_idx[[i % 8 for i in range(k)]],
                    dtype=torch.int32,
                    device=qweight.device,
                )[:, None].expand([-1, N])
                * 4
            )
            dst_data = (data >> shift) & 0xF

            shift_pack = (
                torch.tensor(
                    pack_idx[[i % 8 for i in range(k)]],
                    dtype=torch.int32,
                    device=qweight.device,
                )[:, None].expand([-1, N])
                * 4
            )
            dst_data = dst_data << shift_pack

            for i in range(0, k, 8):
                tmp = dst_data[i, :]
                for j in range(i + 1, i + 8):
                    tmp = torch.bitwise_or(tmp, dst_data[j, :])
                shuffled_weight[e, i // 8, :] = tmp

        return shuffled_weight

    def linear_silu_mul(self, x):
        half = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (half,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        return torch.ops.torch_ipex.silu_and_mul(x, out)

    def linear_swiglu_oai(self, x):
        half = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (half,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        torch.ops.torch_ipex.swigluoai_and_mul(out, x, 1.702, 7.0)
        return out

    def linear_activaion(self, x, activation: Optional[str] = "silu"):
        if activation == "silu":
            return self.linear_silu_mul(x)
        elif activation == "swiglu_oai":
            return self.linear_swiglu_oai(x)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def fused_moe_experts(
        self,
        hidden_states,
        rows_for_experts=None,
        activation: Optional[str] = "silu",
        residual=None,
    ):
        hidden_states = torch.xpu.moe_gemm(
            hidden_states,
            self.W13,
            rows_for_experts,
            self.num_experts,
            self.w13_input_scale_inv,
            self.w13_weight_scale_inv,
            bias=self.w13_bias,
            is_mxfp4=self.is_mxfp4,
            is_fp8=self.is_fp8,
            is_int4=self.is_int4,
            use_native=self.moe_gemm_using_native_impl,
        )
        hidden_states = self.linear_activaion(hidden_states, activation)
        hidden_states = torch.xpu.moe_gemm(
            hidden_states,
            self.W2,
            rows_for_experts,
            self.num_experts,
            self.w2_input_scale_inv,
            self.w2_weight_scale_inv,
            bias=self.w2_bias,
            is_mxfp4=self.is_mxfp4,
            is_fp8=self.is_fp8,
            is_int4=self.is_int4,
            use_native=self.moe_gemm_using_native_impl,
        )
        return hidden_states

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
        scoring_func: Optional[str] = "sigmoid",
        activation: Optional[str] = "silu",
        e_score_correction_bias: Optional[
            torch.Tensor
        ] = None,  # todo, last three arguments not implement here. tmply added for frontend alignment with CPU
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [*, hidden_dim]
            router_logits: [*, num_experts]
        """

        num_tokens, hidden_dim = hidden_states.shape

        if custom_routing_function is not None:
            routing_weights, selected_experts = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=False,  # renormalize will be handled in moe_gather
            )
            routing_weights = routing_weights.to(torch.float)
            selected_experts = selected_experts.to(torch.int32)
        # --------- fusion:  topk softmax  -------------------
        elif not use_grouped_topk:
            routing_weights = torch.empty(
                num_tokens, top_k, dtype=torch.float32, device=router_logits.device
            )
            selected_experts = torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=router_logits.device
            )
            token_expert_indices = torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=router_logits.device
            )
            torch.ops.torch_ipex.topk_softmax(
                routing_weights,
                selected_experts,
                token_expert_indices,
                router_logits,
                False,
            )

        # --------- fusion:  grouped topk  -------------------
        elif use_grouped_topk:
            routing_weights, selected_experts = torch.ops.torch_ipex.grouped_topk(
                hidden_states,
                router_logits,
                top_k,
                False,  # renormalize will be handled in moe_gather
                num_expert_group,
                topk_group,
                scoring_func,
                e_score_correction_bias,
            )
        else:
            raise ValueError(
                "Either custom_routing_function or use_grouped_topk should be set."
            )

        rows_for_experts, expert_offsets = torch.ops.torch_ipex.moe_rows_counts(
            selected_experts, self.experts_start_id, self.num_experts
        )

        # --------- fusion: scatter + moegemm + gather -------------------
        # scatter hidden_states such that the token stride for each expert's input is contiguous
        reordered_hidden_states, mapped_slot = torch.ops.torch_ipex.moe_scatter(
            hidden_states,
            rows_for_experts,
            selected_experts,
            expert_offsets,
            self.experts_start_id,
            self.num_experts,
            top_k,
        )

        reordered_moe_output = self.fused_moe_experts(
            reordered_hidden_states,
            rows_for_experts,
            activation=activation,
        )
        moe_output = torch.ops.torch_ipex.moe_gather(
            reordered_moe_output,
            routing_weights,
            mapped_slot,
            self.num_experts,
            top_k,
            renormalize,
        )
        moe_output = moe_output.reshape(num_tokens, hidden_dim)
        return moe_output
