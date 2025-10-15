# Gated Mixture of Experts (MoE) module designed for VLLM/TGI.
# This code tests the functionality of the MoE module.

import torch
import intel_extension_for_pytorch as ipex  # noqa
import copy
import pytest
import numpy as np

dtype = torch.bfloat16


class MoELayerMxfp4(torch.nn.Module):

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
            torch.randint(
                0,
                0xFF,
                [num_experts, intermediate_size, hidden_size // 2],
                dtype=torch.uint8,
                device="xpu",
            ),
            requires_grad=False,
        )
        torch.manual_seed(45)
        self.w3_weight = torch.nn.Parameter(
            torch.randint(
                0,
                0xFF,
                [num_experts, intermediate_size, hidden_size // 2],
                dtype=torch.uint8,
                device="xpu",
            ),
            requires_grad=False,
        )
        torch.manual_seed(46)
        self.w2_weight = torch.nn.Parameter(
            torch.randint(
                0,
                0xFF,
                [num_experts, hidden_size, intermediate_size // 2],
                dtype=torch.uint8,
                device="xpu",
            ),
            requires_grad=False,
        )

        self.group_size = 32
        self.group_num_w13 = hidden_size // self.group_size
        self.group_num_w2 = intermediate_size // self.group_size

        torch.manual_seed(47)
        self.w13_scale = torch.nn.Parameter(
            torch.randint(
                0,
                0x7F,
                [num_experts, intermediate_size * 2, self.group_num_w13],
                dtype=torch.uint8,
                device="xpu",
            ),
            requires_grad=False,
        )

        torch.manual_seed(48)
        self.w2_scale = torch.nn.Parameter(
            torch.randint(
                0,
                0x7F,
                [num_experts, hidden_size, self.group_num_w2],
                dtype=torch.uint8,
                device="xpu",
            ),
            requires_grad=False,
        )

        self.w1_weight_bf16 = torch.empty(
            num_experts, intermediate_size, hidden_size, dtype=dtype, device="xpu"
        )
        for i in range(num_experts):
            self.w1_weight_bf16[i] = self.dequantize(
                self.w1_weight[i],
                self.w13_scale[i, :intermediate_size],
                self.group_size,
            )

        self.w3_weight_bf16 = torch.empty(
            num_experts, intermediate_size, hidden_size, dtype=dtype, device="xpu"
        )
        for i in range(num_experts):
            self.w3_weight_bf16[i] = self.dequantize(
                self.w3_weight[i],
                self.w13_scale[i, intermediate_size:],
                self.group_size,
            )

        self.w2_weight_bf16 = torch.empty(
            num_experts, hidden_size, intermediate_size, dtype=dtype, device="xpu"
        )
        for i in range(num_experts):
            self.w2_weight_bf16[i] = self.dequantize(
                self.w2_weight[i], self.w2_scale[i], self.group_size
            )

        torch.manual_seed(49)
        self.w13_bias = torch.nn.Parameter(
            torch.randn(num_experts, intermediate_size * 2, dtype=dtype, device="xpu"),
            requires_grad=False,
        )

        torch.manual_seed(50)
        self.w2_bias = torch.nn.Parameter(
            torch.randn(num_experts, hidden_size, dtype=dtype, device="xpu"),
            requires_grad=False,
        )

        self.ipex_moe = ipex.llm.modules.GatedMLPMOE(
            copy.deepcopy(self.w1_weight.view(torch.int32)),
            copy.deepcopy(self.w2_weight.view(torch.int32)),
            copy.deepcopy(self.w3_weight.view(torch.int32)),
            w1_scale_inv=self.w13_scale,
            w2_scale_inv=self.w2_scale,
            w13_bias=self.w13_bias,
            w2_bias=self.w2_bias,
            is_mxfp4=True,
        )

        self.act_fn = torch.nn.SiLU()

    def dequantize(self, qweight, scales, group_size):
        # qweight: [N, K//2]
        n = qweight.shape[0]
        k = qweight.shape[1] * 2
        # use pre-shuffle
        unpack_idx = np.array([0, 1])
        data = qweight[:, [i // 2 for i in range(k)]]
        shift = (
            torch.tensor(
                unpack_idx[[i % 2 for i in range(k)]],
                dtype=torch.int32,
                device=qweight.device,
            )[None, :].expand([n, -1])
            * 4
        )
        dst_data = (data >> shift) & 0xF

        table = torch.tensor(
            [
                0b0000000000000000,
                0b0011111100000000,
                0b0011111110000000,
                0b0011111111000000,
                0b0100000000000000,
                0b0100000001000000,
                0b0100000010000000,
                0b0100000011000000,
                0b1000000000000000,
                0b1011111100000000,
                0b1011111110000000,
                0b1011111111000000,
                0b1100000000000000,
                0b1100000001000000,
                0b1100000010000000,
                0b1100000011000000,
            ],
            dtype=torch.int32,
            device="xpu",
        )
        dst_data = table[dst_data].to(torch.uint16).view(torch.bfloat16)
        expand_scales = scales[:, [i // group_size for i in range(k)]]
        dst_scale = (
            (expand_scales.to(torch.int32) << 7).to(torch.uint16).view(torch.bfloat16)
        )
        weight_bf16 = dst_data * dst_scale

        return weight_bf16

    def ref_mlp(self, hidden_states: torch.Tensor, expert_id: int) -> torch.Tensor:
        w1_out = torch.nn.functional.linear(
            hidden_states, self.w1_weight_bf16[expert_id]
        )
        w1_out += self.w13_bias[expert_id, : self.w1_weight_bf16.shape[1]]
        w1_out = self.act_fn(w1_out)
        w3_out = torch.nn.functional.linear(
            hidden_states, self.w3_weight_bf16[expert_id]
        )
        w3_out += self.w13_bias[expert_id, self.w1_weight_bf16.shape[1] :]
        current_hidden_states = w1_out * w3_out
        current_hidden_states = torch.nn.functional.linear(
            current_hidden_states, self.w2_weight_bf16[expert_id]
        )
        current_hidden_states += self.w2_bias[expert_id]
        return current_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_ipex_api=False,
        use_grouped_topk=False,
        topk_group=None,
        num_expert_group=None,
        scoring_func="sigmoid",
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
                scoring_func=scoring_func,
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
                scoring_func=scoring_func,
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
        scoring_func="sigmoid",
    ) -> torch.Tensor:
        if use_grouped_topk:
            routing_weights, selected_experts = torch.ops.torch_ipex.grouped_topk(
                hidden_states,
                router_logits,
                self.top_k,
                True,
                num_expert_group,
                topk_group,
                scoring_func,
                None,
            )
            selected_experts = selected_experts.to(torch.int64)
        else:
            routing_weights, selected_experts = torch.ops.torch_ipex.topk_softmax(
                router_logits, self.top_k, False
            )
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        num_tokens, hidden_dim = hidden_states.shape
        final_hidden_states = torch.zeros(
            (num_tokens, hidden_dim),
            dtype=torch.float,
            device=hidden_states.device,
        )
        expert_mask = torch.nn.functional.one_hot(
            selected_experts.to(torch.int64), num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                self.ref_mlp(current_state, expert_idx).to(torch.float)
                * routing_weights[top_x, idx, None]
            )
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        return final_hidden_states.to(torch.bfloat16)


class TestMoEModule:
    @pytest.mark.parametrize("num_experts", [8, 16])
    @pytest.mark.parametrize("tokens", [1, 8, 1024])
    @pytest.mark.parametrize("use_grouped_topk", [True, False])
    @pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
    def test(self, num_experts, tokens, use_grouped_topk, scoring_func):
        torch.manual_seed(0)
        top_k = 4
        hidden_size = 1024
        intermediate_size = 1024
        topk_group = None
        num_expert_group = None
        if use_grouped_topk:
            topk_group = 2
            num_expert_group = 2
        moe_module = (
            MoELayerMxfp4(num_experts, top_k, hidden_size, intermediate_size)
            .eval()
            .to(dtype)
        )
        torch.manual_seed(0)
        x = torch.randn([tokens, hidden_size], device="xpu").to(dtype) / 8
        x_ = copy.deepcopy(x)
        ref_out = moe_module(
            x,
            use_ipex_api=False,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
        )
        ipex_out = moe_module(
            x_,
            use_ipex_api=True,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
        )

        torch.testing.assert_close(
            ref_out, ipex_out, atol=1e-1, rtol=1e-1, equal_nan=True
        )
