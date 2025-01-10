import torch
from torch import nn
from typing import Optional


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
    def __init__(self, W13, W2, W3=None, use_prepack=False):
        super().__init__()
        self.num_experts = W2.shape[0]
        if W3 is None:
            self.W13 = torch.transpose(W13, 1, 2).contiguous()
        else:
            self.W13 = torch.transpose(torch.cat((W13, W3), dim=1), 1, 2).contiguous()
        self.W2 = torch.transpose(W2, 1, 2).contiguous()

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
        hidden_states = torch.ops.torch_ipex.moe_gemm(
            hidden_states,
            self.W13,
            rows_for_experts,
            rows_for_experts_cpu,
            self.num_experts,
        )
        hidden_states = self.linear_silu_mul(hidden_states)
        hidden_states = torch.ops.torch_ipex.moe_gemm(
            hidden_states,
            self.W2,
            rows_for_experts,
            rows_for_experts_cpu,
            self.num_experts,
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
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [*, hidden_dim]
            router_logits: [*, num_experts]
        """
        assert not use_grouped_topk
        assert num_expert_group is None
        assert topk_group is None

        num_tokens, hidden_dim = hidden_states.shape
        # --------- fusion:  topk softmax  -------------------
        routing_weights, selected_experts, rows_for_experts, expert_offsets = (
            torch.ops.torch_ipex.topk_softmax(router_logits, top_k)
        )

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
