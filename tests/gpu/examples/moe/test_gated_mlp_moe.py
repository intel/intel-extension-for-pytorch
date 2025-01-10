# Gated Mixture of Experts (MoE) module designed for VLLM/TGI, currently supporting only the Mixtral 8x7B model.
# This code tests the functionality of the MoE module.

import torch
import intel_extension_for_pytorch as ipex
from torch.testing._internal.common_utils import TestCase
import copy


class MixtralMoE(torch.nn.Module):

    def __init__(self, num_experts, top_k, hidden_size, intermediate_size):
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
        self.ipex_moe = ipex.llm.modules.GatedMLPMOE(
            copy.deepcopy(self.w1_weight),
            copy.deepcopy(self.w2_weight),
            copy.deepcopy(self.w3_weight),
        )
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
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        if not use_ipex_api:
            return self.ref_moe_module(hidden_states, router_logits)
        else:
            use_grouped_topk = False
            renormalize = True
            final_hidden_states = self.ipex_moe(
                hidden_states, use_grouped_topk, self.top_k, router_logits, renormalize
            )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        return final_hidden_states

    def ref_moe_module(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=1, dtype=torch.float
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
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


class TestMoEModule(TestCase):
    def test(self):
        num_experts = 8
        top_k = 2
        hidden_size = 2048
        intermediate_size = 14336
        dtype = torch.float16
        moe_module = (
            MixtralMoE(num_experts, top_k, hidden_size, intermediate_size)
            .eval()
            .to(dtype)
        )
        torch.manual_seed(10)
        x = torch.randn([1, hidden_size], device="xpu").to(dtype) / 10
        x_ = copy.deepcopy(x)
        ref_out = moe_module(x)
        ipex_out = moe_module(x_, use_ipex_api=True)
        self.assertEqual(ref_out, ipex_out, atol=1e-2, rtol=1e-2)
