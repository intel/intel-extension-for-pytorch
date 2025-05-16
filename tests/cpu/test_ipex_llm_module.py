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
import itertools
import torch.nn.functional as F

from typing import Optional

try:
    import einops  # noqa: F401

    HAS_EINPOS = True
except ImportError:
    HAS_EINPOS = False

skipIfNoEINPOS = unittest.skipIf(not HAS_EINPOS, "no einops")


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


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
    prev_state=None,
    final_state_out=None,
):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    prev_state: r(B D N), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    from einops import rearrange, repeat

    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate)) if prev_state is None else prev_state
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            else:
                y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            if final_state_out is None:
                final_state_out = x
            else:
                final_state_out.copy_(x)
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, final_state_out)


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
        assert ipex.llm.modules.MambaMixer is not None
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

    def test_causal_conv1d_update(self):
        def causal_conv1d_update_ref(
            x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
        ):
            """
            x: (batch, dim) or (batch, dim, seqlen)
            conv_state: (batch, dim, state_len), where state_len >= width - 1
            weight: (dim, width)
            bias: (dim,)
            cache_seqlens: (batch,), dtype int32.
                If not None, the conv_state is treated as a circular buffer.
                The conv_state will be updated by copying x to the
                conv_state starting at the index
                @cache_seqlens % state_len before performing the convolution.

            out: (batch, dim) or (batch, dim, seqlen)
            """
            if activation not in [None, "silu", "swish"]:
                raise NotImplementedError("activation must be None, silu, or swish")
            dtype_in = x.dtype
            unsqueeze = x.dim() == 2
            if unsqueeze:
                x = x.unsqueeze(-1)
            batch, dim, seqlen = x.shape
            width = weight.shape[1]
            state_len = conv_state.shape[-1]
            assert conv_state.shape == (batch, dim, state_len)
            assert weight.shape == (dim, width)
            if cache_seqlens is None:
                x_new = torch.cat([conv_state, x], dim=-1).to(
                    weight.dtype
                )  # (batch, dim, state_len + seqlen)
                conv_state.copy_(x_new[:, :, -state_len:])
            else:
                width_idx = torch.arange(
                    -(width - 1), 0, dtype=torch.long, device=x.device
                ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
                width_idx = (
                    torch.remainder(width_idx, state_len)
                    .unsqueeze(1)
                    .expand(-1, dim, -1)
                )
                x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(
                    weight.dtype
                )
                copy_idx = torch.arange(
                    seqlen, dtype=torch.long, device=x.device
                ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
                copy_idx = (
                    torch.remainder(copy_idx, state_len)
                    .unsqueeze(1)
                    .expand(-1, dim, -1)
                )
                conv_state.scatter_(2, copy_idx, x)
            out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
                :, :, -seqlen:
            ]
            if unsqueeze:
                out = out.squeeze(-1)
            return (out if activation is None else F.silu(out)).to(dtype=dtype_in)

        def causal_conv1d_update_ipex(
            x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
        ):
            return ipex.llm.modules.MambaMixer.causal_conv1d_update(
                x, conv_state, weight, bias, activation, cache_seqlens
            )

        batch = 2
        seqlens = [1, 2, 3]
        width = 4
        dtypes = [torch.float, torch.bfloat16]
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        has_bias_list = [True, False]
        dims = [2048, 2048 + 16, 4096]
        silu_activation = [True, False]
        state_lens = [width - 1, width, width + 1]
        has_cache_seqlens_list = [True, False]
        for (
            dtype,
            has_bias,
            dim,
            activation,
            state_len,
            seqlen,
            has_cache_seqlens,
        ) in itertools.product(
            dtypes,
            has_bias_list,
            dims,
            silu_activation,
            state_lens,
            seqlens,
            has_cache_seqlens_list,
        ):
            x = torch.rand(batch, dim, seqlen).to(dtype)
            x_ref = x.clone()
            conv_state = torch.rand(batch, dim, state_len).to(dtype)
            weight = torch.rand(dim, width).to(dtype)
            bias = torch.rand(dim).to(dtype) if has_bias else None
            conv_state_ref = conv_state.clone()
            act = None if not activation else "silu"
            cache_seqlens = (
                torch.randint(0, 1024, (batch,), dtype=torch.int32)
                if has_cache_seqlens
                else None
            )

            out_ref = causal_conv1d_update_ref(
                x_ref,
                conv_state_ref,
                weight,
                bias,
                activation=act,
                cache_seqlens=cache_seqlens,
            )
            out_ipex = causal_conv1d_update_ipex(
                x, conv_state, weight, bias, activation=act, cache_seqlens=cache_seqlens
            )
            rtol, atol = (3e-4, 1e-3) if dtype == torch.float32 else (3e-3, 5e-3)
            if dtype == torch.bfloat16:
                rtol, atol = 1e-2, 5e-2
            self.assertTrue(torch.allclose(out_ref, out_ipex, rtol=rtol, atol=atol))
            self.assertEqual(conv_state_ref, conv_state)

    def test_causal_conv1d_fn(self):
        def causal_conv1d_ipex(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            initial_states: Optional[torch.Tensor] = None,
            return_final_states: bool = False,
            final_states_out: Optional[torch.Tensor] = None,
            activation: Optional[str] = "silu",
        ):
            return ipex.llm.modules.MambaMixer.causal_conv1d_fn(
                x,
                weight,
                bias,
                initial_states,
                return_final_states,
                final_states_out,
                activation,
            )

        batch = 1
        seqlens = [1, 8, 16, 32, 64, 128, 256, 512, 784, 1024, 1025, 2048, 4096]
        width = 4
        dtypes = [torch.float, torch.bfloat16]
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        has_bias_list = [True]
        dims = [64]
        silu_activation = [True]
        has_initial_state_list = [True, False]
        for (
            dtype,
            has_bias,
            dim,
            activation,
            seqlen,
            has_initial_state,
        ) in itertools.product(
            dtypes,
            has_bias_list,
            dims,
            silu_activation,
            seqlens,
            has_initial_state_list,
        ):
            x = torch.rand(batch, dim, seqlen).to(dtype)
            x_ref = x.clone()
            weight = torch.rand(dim, width).to(dtype)
            bias = torch.rand(dim).to(dtype) if has_bias else None
            if has_initial_state:
                initial_states = torch.randn(batch, dim, width - 1, dtype=dtype)
            else:
                initial_states = None
            x_ref = x.clone()
            weight_ref = weight.clone()
            bias_ref = bias.clone() if bias is not None else None
            initial_states_ref = (
                initial_states.clone() if initial_states is not None else None
            )
            act = None if not activation else "silu"

            out_ref, final_states_ref = causal_conv1d_ref(
                x_ref,
                weight_ref,
                bias_ref,
                initial_states=initial_states_ref,
                return_final_states=True,
                activation=act,
            )
            out_ipex, final_states_ipex = causal_conv1d_ipex(
                x,
                weight,
                bias,
                initial_states=initial_states,
                return_final_states=True,
                activation=act,
            )
            rtol, atol = (3e-4, 1e-3) if dtype == torch.float32 else (3e-3, 5e-3)
            if dtype == torch.bfloat16:
                rtol, atol = 1e-2, 5e-2
            self.assertEqual(out_ref, out_ipex, rtol=rtol, atol=atol)
            self.assertEqual(final_states_ref, final_states_ipex, rtol=rtol, atol=atol)

    @skipIfNoEINPOS
    def test_selective_state_update(self):
        from einops import rearrange, repeat

        def selective_state_update_ref(
            state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
        ):
            """
            Argument:
                state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
                x: (batch, dim) or (batch, nheads, dim)
                dt: (batch, dim) or (batch, nheads, dim)
                A: (dim, dstate) or (nheads, dim, dstate)
                B: (batch, dstate) or (batch, ngroups, dstate)
                C: (batch, dstate) or (batch, ngroups, dstate)
                D: (dim,) or (nheads, dim)
                z: (batch, dim) or (batch, nheads, dim)
                dt_bias: (dim,) or (nheads, dim)
            Return:
                out: (batch, dim) or (batch, nheads, dim)
            """
            has_heads = state.dim() > 3
            if state.dim() == 3:
                state = state.unsqueeze(1)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            if dt.dim() == 2:
                dt = dt.unsqueeze(1)
            if A.dim() == 2:
                A = A.unsqueeze(0)
            if B.dim() == 2:
                B = B.unsqueeze(1)
            if C.dim() == 2:
                C = C.unsqueeze(1)
            if D is not None and D.dim() == 1:
                D = D.unsqueeze(0)
            if z is not None and z.dim() == 2:
                z = z.unsqueeze(1)
            if dt_bias is not None and dt_bias.dim() == 1:
                dt_bias = dt_bias.unsqueeze(0)
            batch, nheads, dim, dstate = state.shape
            assert x.shape == (batch, nheads, dim)
            assert dt.shape == x.shape
            assert A.shape == (nheads, dim, dstate)
            ngroups = B.shape[1]
            assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
            assert B.shape == (batch, ngroups, dstate)
            assert C.shape == B.shape
            if D is not None:
                assert D.shape == (nheads, dim)
            if z is not None:
                assert z.shape == x.shape
            if dt_bias is not None:
                assert dt_bias.shape == (nheads, dim)
                dt = dt + dt_bias
            dt = F.softplus(dt) if dt_softplus else dt
            dA = torch.exp(
                rearrange(dt, "b h d -> b h d 1") * A
            )  # (batch, nheads, dim, dstate)
            B = repeat(
                B, "b g n -> b (g h) n", h=nheads // ngroups
            )  # (batch, nheads, dstate)
            C = repeat(
                C, "b g n -> b (g h) n", h=nheads // ngroups
            )  # (batch, nheads, dstate)
            dB = rearrange(dt, "b h d -> b h d 1") * rearrange(
                B, "b h n -> b h 1 n"
            )  # (batch, nheads, dim, dstate)
            state.copy_(
                state * dA + dB * rearrange(x, "b h d -> b h d 1")
            )  # (batch, dim, dstate
            out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
            if D is not None:
                out += (x * D).to(out.dtype)
            out = (out if z is None else out * F.silu(z)).to(x.dtype)
            if not has_heads:
                out = out.squeeze(1)
            return out

        def selective_state_update_ipex(
            state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
        ):
            return ipex.llm.modules.MambaMixer.selective_state_update(
                state,
                x,
                dt,
                A,
                B,
                C,
                D,
                z,
                dt_bias,
                dt_softplus,
            )

        dtypes = [torch.float, torch.bfloat16]
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        has_z_list = [True, False]
        dstate_list = [16, 32, 64]
        dims = [2048, 2048 + 16, 4096]
        batch = 1
        for dtype, has_z, dstate, dim in itertools.product(
            dtypes, has_z_list, dstate_list, dims
        ):
            state = torch.randn(batch, dim, dstate, dtype=dtype)
            x = torch.randn(batch, dim, dtype=dtype)
            dt = torch.randn(batch, dim, dtype=dtype)
            dt_bias = torch.rand(dim) - 4.0
            A = -torch.rand(dim, dstate) - 1.0
            B = torch.randn(batch, dstate)
            C = torch.randn(batch, dstate)
            D = torch.randn(dim)
            z = torch.randn_like(x) if has_z else None
            state_ref = state.detach().clone()
            out_ref = selective_state_update_ref(
                state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
            )
            out_ipex = selective_state_update_ipex(
                state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
            )
            rtol, atol = (3e-4, 1e-3) if dtype == torch.float32 else (3e-3, 5e-3)
            if dtype == torch.bfloat16:
                rtol, atol = 1e-2, 5e-2
            self.assertEqual(out_ref, out_ipex, rtol=rtol, atol=atol)
            self.assertEqual(state_ref, state, rtol=rtol, atol=atol)

    @skipIfNoEINPOS
    def test_selective_scan(self):
        def selective_scan_ipex(
            u,
            delta,
            A,
            B,
            C,
            D=None,
            z=None,
            delta_bias=None,
            delta_softplus=False,
            return_last_state=False,
        ):
            return ipex.llm.modules.MambaMixer.selective_scan_fn(
                u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
            )

        dtypes = [torch.float, torch.bfloat16]
        if core.onednn_has_fp16_support():
            dtypes.append(torch.float16)
        seqlens = [128, 256, 512, 1024, 2048, 4096]
        varBC_groups = [1, 2]
        batch_size = 1
        dim = 4
        dstate = 8
        for dtype, seqlen, BC_group in itertools.product(dtypes, seqlens, varBC_groups):
            rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
            if dtype == torch.bfloat16:
                rtol, atol = 3e-2, 5e-2
            rtolw, atolw = (1e-3, 1e-3)
            rtolw = max(rtolw, rtol)
            atolw = max(atolw, atol)
            A = -0.5 * torch.rand(dim, dstate, dtype=torch.float32)
            A_ref = A.clone()
            if BC_group == 1:
                B_shape = [batch_size, dstate, seqlen]
                C_shape = [batch_size, dstate, seqlen]
            else:
                B_shape = [batch_size, BC_group, dstate, seqlen]
                C_shape = [batch_size, BC_group, dstate, seqlen]
            B = torch.randn(B_shape, dtype=dtype)
            B_ref = B.clone()
            C = torch.randn(C_shape, dtype=dtype)
            C_ref = C.clone()
            D = torch.randn(dim, dtype=torch.float32)
            D_ref = D.clone()
            z = torch.randn(batch_size, dim, seqlen, dtype=dtype)
            z_ref = z.clone()
            delta_bias = 0.5 * torch.rand(dim, dtype=torch.float32)
            u = torch.randn(batch_size, dim, seqlen, dtype=dtype)
            u_ref = u.clone()
            delta = 0.5 * torch.rand(batch_size, dim, seqlen, dtype=dtype)
            delta_ref = delta.clone()
            out_ref, state_ref = selective_scan_ref(
                u_ref,
                delta_ref,
                A_ref,
                B_ref,
                C_ref,
                D_ref,
                z=z_ref,
                delta_bias=delta_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            out_ipex, state_ipex = selective_scan_ipex(
                u,
                delta,
                A,
                B,
                C,
                D,
                z=z,
                delta_bias=delta_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            self.assertEqual(out_ref, out_ipex, rtol=rtol, atol=atol)
            self.assertEqual(state_ref, state_ipex, rtol=rtolw, atol=atolw)


if __name__ == "__main__":
    test = unittest.main()
