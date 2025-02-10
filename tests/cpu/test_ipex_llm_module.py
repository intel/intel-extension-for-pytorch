import unittest
import torch
import math
import intel_extension_for_pytorch as ipex
import intel_extension_for_pytorch._C as core
from torch.testing._internal.common_utils import TestCase
import copy
from intel_extension_for_pytorch.cpu._auto_kernel_selection import (
    _enable_tpp,
    _disable_tpp,
)


class Linear_gelu(torch.nn.Module):
    def __init__(self):
        super(Linear_gelu, self).__init__()
        self.linear = torch.nn.Linear(4096, 4096)

    def forward(self, x):
        return torch.nn.functional.gelu(self.linear(x))


class Linear_newgelu(torch.nn.Module):
    def __init__(self):
        super(Linear_newgelu, self).__init__()
        self.linear = torch.nn.Linear(4096, 4096)

    def forward(self, x):
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


class Linear_silu(torch.nn.Module):
    def __init__(self):
        super(Linear_silu, self).__init__()
        self.linear = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x):
        return torch.nn.functional.silu(self.linear(x))


class Linear_relu(torch.nn.Module):
    def __init__(self):
        super(Linear_relu, self).__init__()
        self.linear = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x):
        return torch.nn.functional.relu(self.linear(x))


class linear2_SiluMul(torch.nn.Module):
    def __init__(self):
        super(linear2_SiluMul, self).__init__()
        self.linear_1 = torch.nn.Linear(4096, 4096, bias=False)
        self.linear_2 = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x):
        return torch.nn.functional.silu(self.linear_1(x)) * self.linear_2(x)


class Linear_mul(torch.nn.Module):
    def __init__(self):
        super(Linear_mul, self).__init__()
        self.linear = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x, y):
        return self.linear(x) * y


class Linear_add(torch.nn.Module):
    def __init__(self):
        super(Linear_add, self).__init__()
        self.linear = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x, y):
        return self.linear(x) + y


class linear_SiluMul(torch.nn.Module):
    def __init__(self):
        super(linear_SiluMul, self).__init__()
        self.linear = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x, y):
        return torch.nn.functional.silu(self.linear(x)) * y


class Linear_add_add(torch.nn.Module):
    def __init__(self):
        super(Linear_add_add, self).__init__()
        self.linear = torch.nn.Linear(4096, 4096)

    def forward(self, x, y, z):
        return self.linear(x) + y + z


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _create_inv_freq(rotary_dim, base):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    return inv_freq


def _update_sin_cos_cache(dtype, rotary_dim, base, seqlen):
    inv_freq = _create_inv_freq(rotary_dim, base)
    t = torch.arange(seqlen, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq.to(device=t.device))
    return torch.sin(freqs).to(dtype), torch.cos(freqs).to(dtype)


def get_sin_cos(
    position_ids: torch.Tensor, rotary_dim, base, seqlen: int, dtype: torch.dtype
):
    sin, cos = _update_sin_cos_cache(dtype, rotary_dim, base, seqlen)
    _cos = torch.index_select(cos, 0, position_ids)
    _sin = torch.index_select(sin, 0, position_ids)
    return _sin.unsqueeze(1), _cos.unsqueeze(1)


def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    rotary_dim = cos.shape[-1]
    x1 = x[..., :rotary_dim]
    x2 = x[..., rotary_dim : 2 * rotary_dim]
    c = x1 * cos - x2 * sin
    d = x1 * sin + x2 * cos
    return torch.cat([c, d], dim=-1)


def add_rmsnorm(residual, x, weight, bias, eps, add_back):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = residual + x
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(variance + eps)
    out = out.to(orig_dtype) * weight
    if add_back and residual is not None:
        residual.copy_(x.to(orig_dtype))
    return out


def add_layernorm(residual, x, weight, bias, eps, add_back):
    if residual is None:
        return torch.nn.functional.layer_norm(
            x, [x.size(-1)], weight=weight, bias=bias, eps=eps
        )
    x = residual + x
    out = torch.nn.functional.layer_norm(
        x, [x.size(-1)], weight=weight, bias=bias, eps=eps
    )
    if add_back:
        residual.copy_(x)
    return out


def silu_mul(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor = None):
    if out is None:
        out = torch.empty_like(x)
    out = torch.nn.functional.silu(x) * y
    return out


def gelu_mul(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor = None, approximate="none"
):
    if out is None:
        out = torch.empty_like(x)
    out = torch.nn.functional.gelu(x, approximate=approximate) * y
    return out


class MixtralMoE(torch.nn.Module):

    def __init__(
        self, num_local_experts, num_experts_per_tok, hidden_size, intermediate_size
    ):
        super().__init__()
        self.num_total_experts = num_local_experts
        self.top_k = num_experts_per_tok

        self.gate = torch.nn.Linear(hidden_size, self.num_total_experts, bias=False)
        self.gate.weight = torch.nn.Parameter(
            torch.rand(self.num_total_experts, hidden_size),
            requires_grad=False,
        )
        self.w1_weight = torch.nn.Parameter(
            torch.rand(num_local_experts, intermediate_size, hidden_size),
            requires_grad=False,
        )
        self.w3_weight = torch.nn.Parameter(
            torch.rand(num_local_experts, intermediate_size, hidden_size),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.rand(num_local_experts, hidden_size, intermediate_size),
            requires_grad=False,
        )

        self.ipex_moe = ipex.llm.modules.GatedMLPMOE(
            copy.deepcopy(self.w1_weight),
            copy.deepcopy(self.w2_weight),
            copy.deepcopy(self.w3_weight),
            use_prepack=False,
        )
        self.ipex_moe_with_prepack = ipex.llm.modules.GatedMLPMOE(
            copy.deepcopy(self.w1_weight),
            copy.deepcopy(self.w2_weight),
            copy.deepcopy(self.w3_weight),
            use_prepack=True,
        )
        self.act_fn = torch.nn.SiLU()

    def forward_mlp(self, hidden_states: torch.Tensor, expert_id: int) -> torch.Tensor:
        w1_out = torch.nn.functional.linear(hidden_states, self.w1_weight[expert_id])
        w1_out = self.act_fn(w1_out)
        w3_out = torch.nn.functional.linear(hidden_states, self.w3_weight[expert_id])
        current_hidden_states = w1_out * w3_out
        current_hidden_states = torch.nn.functional.linear(
            current_hidden_states, self.w2_weight[expert_id]
        )
        return current_hidden_states

    def forward(
        self, hidden_states: torch.Tensor, use_ipex_api=False, use_ipex_prepack=False
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        if not use_ipex_api:
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
                selected_experts, num_classes=self.num_total_experts
            ).permute(2, 1, 0)
            for expert_idx in range(self.num_total_experts):
                idx, top_x = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = (
                    self.forward_mlp(current_state, expert_idx)
                    * routing_weights[top_x, idx, None]
                )
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )
        else:
            if use_ipex_prepack:
                final_hidden_states = self.ipex_moe_with_prepack(
                    hidden_states, False, self.top_k, router_logits, True
                )
            else:
                final_hidden_states = self.ipex_moe(
                    hidden_states, False, self.top_k, router_logits, True
                )

        return final_hidden_states.view(num_tokens, hidden_dim)


class Deepseekv2MoE(torch.nn.Module):

    def __init__(
        self, n_routed_experts, num_experts_per_tok, hidden_size, intermediate_size
    ):
        super().__init__()
        self.num_total_experts = n_routed_experts
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts

        self.gate = torch.nn.Linear(hidden_size, self.n_routed_experts, bias=False)
        self.gate.weight = torch.nn.Parameter(
            torch.rand(self.n_routed_experts, hidden_size),
            requires_grad=False,
        )
        self.w1_weight = torch.nn.Parameter(
            torch.rand(n_routed_experts, intermediate_size, hidden_size),
            requires_grad=False,
        )
        self.w3_weight = torch.nn.Parameter(
            torch.rand(n_routed_experts, intermediate_size, hidden_size),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.rand(n_routed_experts, hidden_size, intermediate_size),
            requires_grad=False,
        )

        self.ipex_moe = ipex.llm.modules.GatedMLPMOE(
            copy.deepcopy(self.w1_weight),
            copy.deepcopy(self.w2_weight),
            copy.deepcopy(self.w3_weight),
            use_prepack=False,
        )
        self.ipex_moe_with_prepack = ipex.llm.modules.GatedMLPMOE(
            copy.deepcopy(self.w1_weight),
            copy.deepcopy(self.w2_weight),
            copy.deepcopy(self.w3_weight),
            use_prepack=True,
        )
        self.act_fn = torch.nn.SiLU()
        self.n_group = 8
        self.topk_group = 3

    def forward_mlp(self, hidden_states: torch.Tensor, expert_id: int) -> torch.Tensor:
        w1_out = torch.nn.functional.linear(hidden_states, self.w1_weight[expert_id])
        w1_out = self.act_fn(w1_out)
        w3_out = torch.nn.functional.linear(hidden_states, self.w3_weight[expert_id])
        current_hidden_states = w1_out * w3_out
        current_hidden_states = torch.nn.functional.linear(
            current_hidden_states, self.w2_weight[expert_id]
        )
        return current_hidden_states

    def forward(
        self, hidden_states: torch.Tensor, use_ipex_api=False, use_ipex_prepack=False
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        if not use_ipex_api:
            router_logits = router_logits.softmax(dim=-1, dtype=torch.float32)
            group_scores = (
                router_logits.view(num_tokens, self.n_group, -1).max(dim=-1).values
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(num_tokens, self.n_group, self.n_routed_experts // self.n_group)
                .reshape(num_tokens, -1)
            )  # [n, e]
            tmp_scores = router_logits.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            routing_weights, selected_experts = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )

            routing_weights = routing_weights.to(hidden_states.dtype)
            final_hidden_states = torch.zeros(
                (num_tokens, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.num_total_experts
            ).permute(2, 1, 0)
            for expert_idx in range(self.num_total_experts):
                idx, top_x = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = (
                    self.forward_mlp(current_state, expert_idx)
                    * routing_weights[top_x, idx, None]
                )
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )
        else:
            if use_ipex_prepack:
                final_hidden_states = self.ipex_moe_with_prepack(
                    hidden_states,
                    True,
                    self.top_k,
                    router_logits,
                    False,
                    self.topk_group,
                    self.n_group,
                )
            else:
                final_hidden_states = self.ipex_moe(
                    hidden_states,
                    True,
                    self.top_k,
                    router_logits,
                    False,
                    self.topk_group,
                    self.n_group,
                )

        return final_hidden_states.view(num_tokens, hidden_dim)


class Deepseekv3MoE(torch.nn.Module):

    def __init__(
        self, n_routed_experts, num_experts_per_tok, hidden_size, intermediate_size
    ):
        super().__init__()
        self.num_total_experts = n_routed_experts
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts

        self.gate = torch.nn.Linear(hidden_size, self.n_routed_experts, bias=False)
        self.gate.weight = torch.nn.Parameter(
            torch.rand(self.n_routed_experts, hidden_size),
            requires_grad=False,
        )
        self.w1_weight = torch.nn.Parameter(
            torch.rand(n_routed_experts, intermediate_size, hidden_size),
            requires_grad=False,
        )
        self.w3_weight = torch.nn.Parameter(
            torch.rand(n_routed_experts, intermediate_size, hidden_size),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.rand(n_routed_experts, hidden_size, intermediate_size),
            requires_grad=False,
        )

        self.ipex_moe = ipex.llm.modules.GatedMLPMOE(
            copy.deepcopy(self.w1_weight),
            copy.deepcopy(self.w2_weight),
            copy.deepcopy(self.w3_weight),
            use_prepack=False,
        )
        self.ipex_moe_with_prepack = ipex.llm.modules.GatedMLPMOE(
            copy.deepcopy(self.w1_weight),
            copy.deepcopy(self.w2_weight),
            copy.deepcopy(self.w3_weight),
            use_prepack=True,
        )
        self.act_fn = torch.nn.SiLU()
        self.n_group = 8
        self.topk_group = 4
        self.e_score_correction_bias = torch.nn.Parameter(
            torch.ones((n_routed_experts))
        )

    def forward_mlp(self, hidden_states: torch.Tensor, expert_id: int) -> torch.Tensor:
        w1_out = torch.nn.functional.linear(hidden_states, self.w1_weight[expert_id])
        w1_out = self.act_fn(w1_out)
        w3_out = torch.nn.functional.linear(hidden_states, self.w3_weight[expert_id])
        current_hidden_states = w1_out * w3_out
        current_hidden_states = torch.nn.functional.linear(
            current_hidden_states, self.w2_weight[expert_id]
        )
        return current_hidden_states

    def forward(
        self, hidden_states: torch.Tensor, use_ipex_api=False, use_ipex_prepack=False
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        if not use_ipex_api:
            scores = router_logits.sigmoid()
            scores_for_choice = scores.view(
                num_tokens, -1
            ) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(num_tokens, self.n_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(num_tokens, self.n_group, self.n_routed_experts // self.n_group)
                .reshape(num_tokens, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(
                ~score_mask.bool(), 0.0
            )  # [n, e]
            _, selected_experts = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
            routing_weights = scores.gather(1, selected_experts)

            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(hidden_states.dtype)
            final_hidden_states = torch.zeros(
                (num_tokens, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.num_total_experts
            ).permute(2, 1, 0)
            for expert_idx in range(self.num_total_experts):
                idx, top_x = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = (
                    self.forward_mlp(current_state, expert_idx)
                    * routing_weights[top_x, idx, None]
                )
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )
        else:
            if use_ipex_prepack:
                final_hidden_states = self.ipex_moe_with_prepack(
                    hidden_states,
                    True,
                    self.top_k,
                    router_logits,
                    True,
                    self.topk_group,
                    self.n_group,
                    scoring_func="sigmoid",
                    e_score_correction_bias=self.e_score_correction_bias,
                )
            else:
                final_hidden_states = self.ipex_moe(
                    hidden_states,
                    True,
                    self.top_k,
                    router_logits,
                    True,
                    self.topk_group,
                    self.n_group,
                    scoring_func="sigmoid",
                    e_score_correction_bias=self.e_score_correction_bias,
                )

        return final_hidden_states.view(num_tokens, hidden_dim)


class TestLLMModules(TestCase):
    def test_linearfusion_args0(self):
        x1 = torch.rand(1, 4, 4096)
        x2 = copy.deepcopy(x1)
        ref_scope = [
            Linear_silu,
            Linear_gelu,
            Linear_newgelu,
            Linear_relu,
            linear2_SiluMul,
        ]
        ipex_scope = [
            ipex.llm.modules.LinearSilu,
            ipex.llm.modules.LinearGelu,
            ipex.llm.modules.LinearNewGelu,
            ipex.llm.modules.LinearRelu,
            ipex.llm.modules.Linear2SiluMul,
        ]
        dtypes = [
            torch.float32,
        ]
        if core.onednn_has_bf16_support():
            dtypes.append(torch.bfloat16)
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        with torch.no_grad():
            for i in range(len(ref_scope)):
                for dtype in dtypes:
                    for use_ipex_optimize in [True, False]:
                        for use_tpp in [True, False]:
                            model = ref_scope[i]().eval().to(dtype)
                            ref_out = model(x1.to(dtype))
                            if use_ipex_optimize:
                                if use_tpp:
                                    if dtype in [torch.bfloat16, torch.float16]:
                                        _enable_tpp()
                                    else:
                                        continue
                                model = ipex.optimize(model, dtype=dtype)
                            else:
                                if use_tpp:
                                    continue
                            if ipex_scope[i] != ipex.llm.modules.Linear2SiluMul:
                                model = ipex_scope[i](model.linear)
                            else:
                                model = ipex_scope[i](model.linear_1, model.linear_2)
                            out = model(x2.to(dtype))
                            atol = None
                            rtol = None
                            if dtype is torch.float16:
                                atol = 1e-3
                                rtol = 1e-3
                            elif dtype is torch.bfloat16:
                                atol = 1e-3
                                rtol = 0.016
                            self.assertEqual(out, ref_out, atol=atol, rtol=rtol)
                            _disable_tpp()

    def test_linearfusion_args1(self):
        x1 = torch.rand(1, 4, 4096)
        x2 = copy.deepcopy(x1)
        ref_scope = [Linear_mul, Linear_add, linear_SiluMul]
        ipex_scope = [
            ipex.llm.modules.LinearMul,
            ipex.llm.modules.LinearAdd,
            ipex.llm.modules.LinearSiluMul,
        ]
        dtypes = [
            torch.float32,
        ]
        if core.onednn_has_bf16_support():
            dtypes.append(torch.bfloat16)
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        with torch.no_grad():
            for i in range(len(ref_scope)):
                for dtype in dtypes:
                    for use_ipex_optimize in [True, False]:
                        for use_tpp in [True, False]:
                            model = ref_scope[i]().eval().to(dtype)
                            ref_out = model(x1.to(dtype), x1.to(dtype))
                            if use_ipex_optimize:
                                if use_tpp:
                                    if dtype in [torch.bfloat16, torch.float16]:
                                        _enable_tpp()
                                    else:
                                        continue
                                model = ipex.optimize(model, dtype=dtype)
                            else:
                                if use_tpp:
                                    continue

                            model = ipex_scope[i](model.linear)

                            out = model(x2.to(dtype), x2.to(dtype))
                            atol = None
                            rtol = None
                            if dtype is torch.float16:
                                atol = 1e-3
                                rtol = 1e-3
                            self.assertEqual(out, ref_out, atol=atol, rtol=rtol)
                            _disable_tpp()

    def test_linearfusion_args2(self):
        x1 = torch.rand(1, 4, 4096)
        x2 = copy.deepcopy(x1)
        ref_scope = [Linear_add_add]
        ipex_scope = [ipex.llm.modules.LinearAddAdd]
        dtypes = [
            torch.float32,
        ]
        if core.onednn_has_bf16_support():
            dtypes.append(torch.bfloat16)
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        with torch.no_grad():
            for i in range(len(ref_scope)):
                for dtype in dtypes:
                    for use_ipex_optimize in [True, False]:
                        for use_tpp in [True, False]:
                            model = ref_scope[i]().eval().to(dtype)
                            ref_out = model(x1.to(dtype), x1.to(dtype), x1.to(dtype))
                            if use_ipex_optimize:
                                if use_tpp:
                                    if dtype in [torch.bfloat16, torch.float16]:
                                        _enable_tpp()
                                    else:
                                        continue
                                model = ipex.optimize(model, dtype=dtype)
                            else:
                                if use_tpp:
                                    continue

                            model = ipex_scope[i](model.linear)

                            out = model(x2.to(dtype), x2.to(dtype), x2.to(dtype))
                            atol = None
                            rtol = None
                            if dtype is torch.float16:
                                atol = 1e-4
                                rtol = 1e-3
                            self.assertEqual(out, ref_out, atol=atol, rtol=rtol)
                            _disable_tpp()

    def test_rmsnorm(self):
        x1 = torch.rand(1, 4, 4096)
        x2 = copy.deepcopy(x1)
        ref_m = LlamaRMSNorm(4096)
        target_m = ipex.llm.modules.RMSNorm(4096)
        dtypes = [torch.float32, torch.bfloat16]
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        for dtype in dtypes:
            ref_m = LlamaRMSNorm(4096).eval().to(dtype)
            target_m = ipex.llm.modules.RMSNorm(4096).to(dtype)
            ref_out = ref_m(x1.to(dtype))
            out = target_m(x2.to(dtype))
            out_2 = ipex.llm.functional.rms_norm(
                x2.to(dtype), ref_m.weight, ref_m.variance_epsilon
            )
            self.assertEqual(out, ref_out)
            self.assertEqual(out_2, ref_out)

    def test_modules_naming(self):
        # below ipex.llm modeules has thier own UTs, here only test their access of naming from ipex.llm.modules
        assert ipex.llm.modules.RotaryEmbedding is not None
        assert ipex.llm.modules.RotaryEmbedding.apply_function is not None
        assert ipex.llm.modules.PagedAttention is not None
        assert ipex.llm.modules.IndirectAccessKVCacheAttention is not None
        assert (
            ipex.llm.modules.IndirectAccessKVCacheAttention.apply_function is not None
        )
        assert ipex.llm.modules.VarlenAttention is not None
        assert ipex.llm.modules.VarlenAttention.apply_function is not None
        assert ipex.llm.modules.FastLayerNorm is not None
        assert ipex.llm.modules.FastLayerNorm.apply_function is not None
        assert ipex.llm.modules.RMSNorm is not None
        assert ipex.llm.modules.RMSNorm.apply_function is not None
        # below only test their access of naming from ipex.llm functional
        assert ipex.llm.functional.rotary_embedding is not None
        assert ipex.llm.functional.rms_norm is not None
        assert ipex.llm.functional.fast_layer_norm is not None
        assert ipex.llm.functional.indirect_access_kv_cache_attention is not None
        assert ipex.llm.functional.varlen_attention is not None

    def test_rotary_embedding_tgi(self):
        test_tensor_size = [
            (1, 32, 128),
            (32, 32, 128),
        ]
        for size in test_tensor_size:
            q = torch.randn(size).float()
            k = torch.randn(size).float()
            rotary_dim = size[-1]
            seqlen = size[0]
            position_ids = torch.arange(size[0])
            sin, cos = get_sin_cos(position_ids, rotary_dim, 10000, seqlen, q.dtype)

            ref_q = apply(q, cos, sin)
            ref_k = apply(k, cos, sin)

            ipex_q, ipex_k = ipex.llm.functional.rotary_embedding(
                q, k, sin, cos, rotary_dim, True
            )

            self.assertEqual(ipex_q, ref_q)
            self.assertEqual(ref_k, ipex_k)

    def test_add_layernorm(self):
        for add_back in [True, False]:
            for dtype in [torch.float, torch.bfloat16, torch.float16]:
                for residual_is_none in [True, False]:
                    weight = torch.nn.Parameter(torch.randn(4096)).to(dtype)
                    eps = 1e-6
                    x = torch.rand(1, 32, 4096).to(dtype)
                    if residual_is_none:
                        residual = None
                    else:
                        if add_back:
                            target_residual = x + x
                        residual = x
                    x_ = copy.deepcopy(x)
                    residual_ = x_ if not residual_is_none else None
                    ref_out = add_layernorm(residual_, x_, weight, None, eps, add_back)
                    ipex_out = ipex.llm.functional.add_layer_norm(
                        residual, x, weight, None, eps, add_back
                    )
                    if not residual_is_none:
                        if add_back:
                            self.assertEqual(residual, target_residual)
                            self.assertEqual(residual_, target_residual)
                        else:
                            self.assertEqual(residual, x)
                            self.assertEqual(residual_, x)
                    self.assertEqual(ref_out, ipex_out)

    def test_add_rmsnorm(self):
        for add_back in [True, False]:
            for dtype in [torch.float, torch.bfloat16, torch.float16]:
                for residual_is_none in [True, False]:
                    weight = torch.nn.Parameter(torch.randn(4096)).to(dtype)
                    eps = 1e-6
                    x = torch.rand(1, 32, 4096).to(dtype)
                    if residual_is_none:
                        residual = None
                    else:
                        if add_back:
                            target_residual = x + x
                        residual = x
                    x_ = copy.deepcopy(x)
                    residual_ = x_ if not residual_is_none else None
                    ref_out = add_rmsnorm(residual_, x_, weight, None, eps, add_back)
                    ipex_out = ipex.llm.functional.add_rms_norm(
                        residual, x, weight, None, eps, add_back
                    )
                    if not residual_is_none:
                        if add_back:
                            self.assertEqual(residual, target_residual)
                            self.assertEqual(residual_, target_residual)
                        else:
                            self.assertEqual(residual, x)
                            self.assertEqual(residual_, x)
                    self.assertEqual(ref_out, ipex_out)

    def test_gelu_mul(self):
        for dtype in [torch.float, torch.bfloat16, torch.float16]:
            for approximate in ["tanh", "none"]:
                x = torch.rand(1, 32, 4096).to(dtype)
                x_ = copy.deepcopy(x)
                ref_out = gelu_mul(x_, x_, approximate=approximate)
                ipex_out = ipex.llm.functional.gelu_mul(x_, x_, approximate=approximate)
                self.assertEqual(ref_out, ipex_out)

    def test_silu_mul(self):
        for dtype in [torch.float, torch.bfloat16, torch.float16]:
            x = torch.rand(1, 32, 4096).to(dtype)
            x_ = copy.deepcopy(x)
            ref_out = silu_mul(x_, x_)
            ipex_out = ipex.llm.functional.silu_mul(x_, x_)
            self.assertEqual(ref_out, ipex_out)

    def test_moe_fusion(self):
        dtypes = [
            torch.float,
        ]
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        moe_modules = [MixtralMoE, Deepseekv2MoE, Deepseekv3MoE]
        with torch.no_grad():
            for moe_id in range(3):
                for dtype in dtypes:
                    for prepack in [True, False]:
                        if moe_id == 0:
                            moe_module = (
                                moe_modules[moe_id](4, 2, 1024, 4096).eval().to(dtype)
                            )
                            x = torch.rand(1, 1024).to(dtype)
                        elif moe_id == 1:
                            moe_module = (
                                moe_modules[moe_id](16, 6, 5120, 4096).eval().to(dtype)
                            )
                            x = torch.rand(1, 5120).to(dtype)
                        elif moe_id == 2:
                            moe_module = (
                                moe_modules[moe_id](16, 8, 7168, 4096).eval().to(dtype)
                            )
                            x = torch.rand(1, 7168).to(dtype)
                        x_ = copy.deepcopy(x)
                        ref_out = moe_module(x)
                        ipex_out = moe_module(
                            x_, use_ipex_api=True, use_ipex_prepack=prepack
                        )
                        self.assertEqual(ref_out, ipex_out)


if __name__ == "__main__":
    test = unittest.main()
