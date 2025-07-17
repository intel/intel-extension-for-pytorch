import torch
from torch import nn
from typing import Optional, Callable


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
    ):
        super().__init__()
        self.num_experts = W2.shape[0]
        if W3 is None:
            self.W13 = torch.transpose(W13, 1, 2)
        else:
            self.W13 = torch.transpose(torch.cat((W13, W3), dim=1), 1, 2)
        self.W2 = torch.transpose(W2, 1, 2)

        if self.W13.dtype is torch.float8_e5m2 or self.W13.dtype is torch.float8_e4m3fn:
            self.W2 = self._weight_for_vnni(self.W2)
            self.W13 = self._weight_for_vnni(self.W13)

        self.W2 = self.W2.contiguous()
        self.W13 = self.W13.contiguous()

        # delete original weight to avoid memory pressure
        W13.data = self.W13
        W2.data = self.W2
        if W3 is not None:
            W3.data = self.W13

        self.w13_weight_scale_inv = w1_scale_inv if w1_scale_inv is not None else None
        self.w2_weight_scale_inv = w2_scale_inv if w2_scale_inv is not None else None
        self.w13_input_scale_inv = a1_scale_inv if a1_scale_inv is not None else None
        self.w2_input_scale_inv = a2_scale_inv if a2_scale_inv is not None else None

        torch.xpu.empty_cache()

    def _weight_for_vnni(self, weight):
        """

        VNNI transform:
        [0,4,8,12,       [0,1,4,5,
        1,5,9,13,   -->   8,9,12,13,
        2,6,10,14,        2,3,6,7,
        3,7,11,15]        10,11,14,15]

        Because xetla read data by tile load,
        VNNI transform in xetla uses two steps:
        [0,4,8,12,       [0,1,2,3,            [0,1,4,5,
        1,5,9,13,   -->   4,5,6,7,     -->     8,9,12,13,
        2,6,10,14,        8,9,10,11,           2,3,6,7,
        3,7,11,15]        12,13,14,15]         10,11,14,15]

        This function reformat weight to avoid second step instead of directly transforming to VNNI.
        And only support activation dtype is bf16/fp16, weight dtype is fp8_e5m2 or fp8_e4m3fn.

        Args:
            weight (torch.Tensor): [experts, K, N].
        Returns:
            torch.Tensor: [experts, K, N]
        """

        E, K, N = weight.shape
        vnni_row, block_size_x = 4, 16
        stride = 2

        assert K % vnni_row == 0, "K should be divisible by vnni_row"
        assert N % block_size_x == 0, "N should be divisible by block_size_x"

        weight = weight.reshape(
            E, K // vnni_row, vnni_row, N // block_size_x, block_size_x
        ).transpose(2, 3)
        weight = weight.reshape(
            E,
            K // vnni_row,
            N // block_size_x,
            vnni_row // stride,
            stride,
            block_size_x,
            1,
        ).transpose(4, 5)
        weight = weight.reshape(
            E,
            K // vnni_row,
            N // block_size_x,
            vnni_row // stride * block_size_x,
            stride,
        )

        weight = torch.cat([weight[..., i::stride, :] for i in range(stride)], dim=-2)

        weight = weight.reshape(
            E,
            K // vnni_row,
            N // block_size_x,
            vnni_row // stride,
            block_size_x,
            stride,
            1,
        ).transpose(4, 5)
        weight = weight.reshape(
            E, K // vnni_row, N // block_size_x, vnni_row, block_size_x
        ).transpose(2, 3)
        weight = weight.reshape(E, K, N)
        return weight

    def linear_silu_mul(self, x):
        half = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (half,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        return torch.ops.torch_ipex.silu_and_mul(x, out)

    def fused_moe_experts(
        self,
        hidden_states,
        rows_for_experts=None,
        rows_for_experts_cpu=None,
        residual=None,
    ):
        hidden_states = torch.xpu.moe_gemm(
            hidden_states,
            self.W13,
            rows_for_experts,
            rows_for_experts_cpu,
            self.num_experts,
            self.w13_input_scale_inv,
            self.w13_weight_scale_inv,
            True,  # True means the input is already ready for VNNI format
        )
        hidden_states = self.linear_silu_mul(hidden_states)
        hidden_states = torch.xpu.moe_gemm(
            hidden_states,
            self.W2,
            rows_for_experts,
            rows_for_experts_cpu,
            self.num_experts,
            self.w2_input_scale_inv,
            self.w2_weight_scale_inv,
            True,  # True means the input is already ready for VNNI format
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
        # --------- fusion:  topk softmax  -------------------
        if not use_grouped_topk:
            routing_weights, selected_experts, rows_for_experts, expert_offsets = (
                torch.ops.torch_ipex.topk_softmax(router_logits, top_k, False)
            )

        # --------- fusion:  grouped topk  -------------------
        elif use_grouped_topk:
            routing_weights, selected_experts, rows_for_experts, expert_offsets = (
                torch.ops.torch_ipex.grouped_topk(
                    hidden_states,
                    router_logits,
                    top_k,
                    False,  # renormalize will be handled in moe_gather
                    num_expert_group,
                    topk_group,
                    scoring_func,
                    e_score_correction_bias,
                )
            )

        if custom_routing_function is not None:
            # Ipex doesn't have op that can calculate rows_for_experts, expert_offsets.
            # Different funcs should have same TopK and different softmax methods,
            # so here uses softmax weights from custom_routing_function
            # and selected_experts, rows_for_experts, expert_offsets from routine topk
            routing_weights, _ = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=False,  # renormalize will be handled in moe_gather
            )
            routing_weights = routing_weights.to(torch.float)

        # --------- fusion: scatter + moegemm + gather -------------------
        # scatter hidden_states such that the token stride for each expert's input is contiguous
        reordered_hidden_states, mapped_slot = torch.ops.torch_ipex.moe_scatter(
            hidden_states,
            rows_for_experts,
            selected_experts,
            expert_offsets,
            self.num_experts,
            top_k,
        )

        rows_for_experts_cpu = rows_for_experts.to("cpu")
        reordered_moe_output = self.fused_moe_experts(
            reordered_hidden_states,
            rows_for_experts,
            rows_for_experts_cpu,
        )
        moe_output = torch.ops.torch_ipex.moe_gather(
            reordered_moe_output,
            routing_weights,
            mapped_slot,
            rows_for_experts,
            self.num_experts,
            top_k,
            renormalize,
        )
        moe_output = moe_output.reshape(num_tokens, hidden_dim)
        return moe_output
