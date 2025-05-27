# Gated Mixture of Experts (MoE) module designed for VLLM/TGI, currently supporting only the Mixtral 8x7B model.
# This code tests the functionality of the MoE module.

import torch
import intel_extension_for_pytorch as ipex  # noqa
import copy
import pytest


class MixtralMoE(torch.nn.Module):

    def __init__(self, num_experts, top_k, hidden_size, intermediate_size, is_fp8):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = torch.nn.Linear(hidden_size, self.num_experts, bias=False)
        torch.manual_seed(42)
        self.gate.weight = torch.nn.Parameter(
            torch.randn((self.num_experts, hidden_size), device="xpu") / 10,
            requires_grad=False,
        )
        torch.manual_seed(43)
        self.w1_weight = torch.nn.Parameter(
            torch.randn((num_experts, intermediate_size, hidden_size), device="xpu")
            / 10,
            requires_grad=False,
        )
        torch.manual_seed(45)
        self.w3_weight = torch.nn.Parameter(
            torch.randn((num_experts, intermediate_size, hidden_size), device="xpu")
            / 10,
            requires_grad=False,
        )
        torch.manual_seed(46)
        self.w2_weight = torch.nn.Parameter(
            torch.randn((num_experts, hidden_size, intermediate_size), device="xpu")
            / 10,
            requires_grad=False,
        )
        if not is_fp8:
            self.ipex_moe = ipex.llm.modules.GatedMLPMOE(
                copy.deepcopy(self.w1_weight),
                copy.deepcopy(self.w2_weight),
                copy.deepcopy(self.w3_weight),
            )
        else:
            dtype = self.w1_weight.dtype
            fp8_dtype = torch.float8_e5m2
            w1_scale = torch.full((num_experts,), 4.0, device="xpu")
            w1_scale_inv = torch.full((num_experts,), 0.25, device="xpu")
            w2_scale = torch.full((num_experts,), 2.0, device="xpu")
            w2_scale_inv = torch.full((num_experts,), 0.5, device="xpu")

            w13_weight = torch.cat((self.w1_weight, self.w3_weight), dim=1)
            w13_weight_fp8 = torch.empty_like(w13_weight, device="xpu", dtype=fp8_dtype)

            w1_weight_fp8 = torch.empty_like(
                self.w1_weight, device="xpu", dtype=fp8_dtype
            )
            w2_weight_fp8 = torch.empty_like(
                self.w2_weight, device="xpu", dtype=fp8_dtype
            )
            w3_weight_fp8 = torch.empty_like(
                self.w3_weight, device="xpu", dtype=fp8_dtype
            )

            for i in range(num_experts):
                w13_weight_fp8[i], _ = torch.ops.torch_ipex.cast_to_fp8(
                    w13_weight[i], w1_scale[i], False, False, fp8_dtype, None
                )

                w1_weight_fp8[i], _ = torch.ops.torch_ipex.cast_to_fp8(
                    self.w1_weight[i], w1_scale[i], False, False, fp8_dtype, None
                )
                w2_weight_fp8[i], _ = torch.ops.torch_ipex.cast_to_fp8(
                    self.w2_weight[i], w2_scale[i], False, False, fp8_dtype, None
                )
                w3_weight_fp8[i], _ = torch.ops.torch_ipex.cast_to_fp8(
                    self.w3_weight[i], w1_scale[i], False, False, fp8_dtype, None
                )

                self.w1_weight[i] = torch.ops.torch_ipex.cast_from_fp8(
                    w1_weight_fp8[i], w1_scale_inv[i], dtype
                )
                self.w2_weight[i] = torch.ops.torch_ipex.cast_from_fp8(
                    w2_weight_fp8[i], w2_scale_inv[i], dtype
                )
                self.w3_weight[i] = torch.ops.torch_ipex.cast_from_fp8(
                    w3_weight_fp8[i], w1_scale_inv[i], dtype
                )

            self.ipex_moe = ipex.llm.modules.GatedMLPMOE(
                copy.deepcopy(w13_weight_fp8),
                copy.deepcopy(w2_weight_fp8),
                W3=None,
                w1_scale_inv=copy.deepcopy(w1_scale_inv),
                w2_scale_inv=copy.deepcopy(w2_scale_inv),
            )

            # XPU does not support cat fp8, W3 should be None
            # Once XPU support cat fp8, use below code
            # self.ipex_moe = ipex.llm.modules.GatedMLPMOE(
            #     copy.deepcopy(w1_weight_fp8),
            #     copy.deepcopy(w2_weight_fp8),
            #     copy.deepcopy(w3_weight_fp8),
            #     w1_scale_inv=copy.deepcopy(w1_scale_inv),
            #     w2_scale_inv=copy.deepcopy(w2_scale_inv),
            # )

        self.act_fn = torch.nn.SiLU()

    def ref_mlp(self, hidden_states: torch.Tensor, expert_id: int) -> torch.Tensor:
        w1_out = torch.nn.functional.linear(hidden_states, self.w1_weight[expert_id])
        w1_out = self.act_fn(w1_out)
        w3_out = torch.nn.functional.linear(hidden_states, self.w3_weight[expert_id])
        current_hidden_states = w1_out * w3_out
        current_hidden_states = torch.nn.functional.linear(
            current_hidden_states, self.w2_weight[expert_id]
        )
        return current_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_ipex_api=False,
        use_grouped_topk=False,
        topk_group=None,
        num_expert_group=None,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        if not use_ipex_api:
            return self.ref_moe_module(
                hidden_states,
                router_logits,
                use_grouped_topk,
                topk_group,
                num_expert_group,
            )
        else:
            renormalize = True
            final_hidden_states = self.ipex_moe(
                hidden_states,
                use_grouped_topk,
                self.top_k,
                router_logits,
                renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
            )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        return final_hidden_states

    def ref_moe_module(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        use_grouped_topk=False,
        topk_group=None,
        num_expert_group=None,
    ) -> torch.Tensor:
        if use_grouped_topk:
            routing_weights, selected_experts, _, _ = (
                torch.ops.torch_ipex.grouped_topk_sigmoid(
                    hidden_states,
                    router_logits,
                    self.top_k,
                    True,
                    num_expert_group,
                    topk_group,
                    "sigmoid",
                    None,
                    True,
                )
            )
            selected_experts = selected_experts.to(torch.int64)
        else:
            routing_weights = torch.nn.functional.softmax(
                router_logits, dim=1, dtype=torch.float
            )
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        num_tokens, hidden_dim = hidden_states.shape
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = torch.zeros(
            (num_tokens, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                self.ref_mlp(current_state, expert_idx)
                * routing_weights[top_x, idx, None]
            )
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        return final_hidden_states


class TestMoEModule:
    @pytest.mark.parametrize("num_experts", [8, 16])
    @pytest.mark.parametrize("use_grouped_topk", [True, False])
    @pytest.mark.parametrize("is_fp8", [True, False])
    def test(self, num_experts, use_grouped_topk, is_fp8):
        top_k = 2
        hidden_size = 1024
        intermediate_size = 14336
        dtype = torch.float16
        topk_group = None
        num_expert_group = None
        if use_grouped_topk:
            topk_group = 2
            num_expert_group = 2
        moe_module = (
            MixtralMoE(num_experts, top_k, hidden_size, intermediate_size, is_fp8)
            .eval()
            .to(dtype)
        )
        torch.manual_seed(10)
        x = torch.randn([1, hidden_size], device="xpu").to(dtype) / 10
        x_ = copy.deepcopy(x)
        ref_out = moe_module(
            x,
            use_ipex_api=False,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
        )
        ipex_out = moe_module(
            x_,
            use_ipex_api=True,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
        )
        torch.testing.assert_close(ref_out, ipex_out, atol=1e-2, rtol=1e-2)
