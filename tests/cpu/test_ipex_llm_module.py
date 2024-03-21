import unittest
import torch
import math
import intel_extension_for_pytorch as ipex
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
        with torch.no_grad():
            for i in range(len(ref_scope)):
                for dtype in [torch.float32, torch.bfloat16]:
                    for use_ipex_optimize in [True, False]:
                        for use_tpp in [True, False]:
                            model = ref_scope[i]().eval().to(dtype)
                            ref_out = model(x1.to(dtype))
                            if use_ipex_optimize:
                                if use_tpp:
                                    if dtype == torch.bfloat16:
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
                            self.assertEqual(out, ref_out)
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
        with torch.no_grad():
            for i in range(len(ref_scope)):
                for dtype in [torch.float32, torch.bfloat16]:
                    for use_ipex_optimize in [True, False]:
                        for use_tpp in [True, False]:
                            model = ref_scope[i]().eval().to(dtype)
                            ref_out = model(x1.to(dtype), x1.to(dtype))
                            if use_ipex_optimize:
                                if use_tpp:
                                    if dtype == torch.bfloat16:
                                        _enable_tpp()
                                    else:
                                        continue
                                model = ipex.optimize(model, dtype=dtype)
                            else:
                                if use_tpp:
                                    continue

                            model = ipex_scope[i](model.linear)

                            out = model(x2.to(dtype), x2.to(dtype))
                            self.assertEqual(out, ref_out)
                            _disable_tpp()

    def test_linearfusion_args2(self):
        x1 = torch.rand(1, 4, 4096)
        x2 = copy.deepcopy(x1)
        ref_scope = [Linear_add_add]
        ipex_scope = [ipex.llm.modules.LinearAddAdd]
        with torch.no_grad():
            for i in range(len(ref_scope)):
                for dtype in [torch.float32, torch.bfloat16]:
                    for use_ipex_optimize in [True, False]:
                        for use_tpp in [True, False]:
                            model = ref_scope[i]().eval().to(dtype)
                            ref_out = model(x1.to(dtype), x1.to(dtype), x1.to(dtype))
                            if use_ipex_optimize:
                                if use_tpp:
                                    if dtype == torch.bfloat16:
                                        _enable_tpp()
                                    else:
                                        continue
                                model = ipex.optimize(model, dtype=dtype)
                            else:
                                if use_tpp:
                                    continue

                            model = ipex_scope[i](model.linear)

                            out = model(x2.to(dtype), x2.to(dtype), x2.to(dtype))
                            self.assertEqual(out, ref_out)
                            _disable_tpp()

    def test_rmsnorm(self):
        x1 = torch.rand(1, 4, 4096)
        x2 = copy.deepcopy(x1)
        ref_m = LlamaRMSNorm(4096)
        target_m = ipex.llm.modules.RMSNorm(4096)
        for dtype in [torch.float32, torch.bfloat16]:
            ref_m = LlamaRMSNorm(4096).eval().to(dtype)
            target_m = ipex.llm.modules.RMSNorm(4096).to(dtype)
            ref_out = ref_m(x1.to(dtype))
            out = target_m(x2.to(dtype))
            out_2 = ipex.llm.modules.RMSNorm.apply(
                x2.to(dtype), ref_m.weight, ref_m.variance_epsilon
            )
            self.assertEqual(out, ref_out)
            self.assertEqual(out_2, ref_out)

    def test_modules_naming(self):
        # below ipex.llm modeules has thier own UTs, here only test their access of naming from ipex.llm.modules
        assert ipex.llm.modules.RotaryEmbedding is not None
        assert ipex.llm.modules.RotaryEmbedding.apply is not None
        assert ipex.llm.modules.PagedAttention is not None
        assert ipex.llm.modules.IndirectAccessKVCache is not None
        assert ipex.llm.modules.IndirectAccessKVCache.apply is not None
        assert ipex.llm.modules.VarlenAttention is not None
        assert ipex.llm.modules.VarlenAttention.apply is not None
        assert ipex.llm.modules.FastLayerNorm is not None
        assert ipex.llm.modules.FastLayerNorm.apply is not None
        assert ipex.llm.modules.RMSNorm is not None
        assert ipex.llm.modules.RMSNorm.apply is not None

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

            ipex_q, ipex_k = ipex.llm.modules.RotaryEmbedding.apply(
                q, k, sin, cos, rotary_dim, True
            )

            self.assertEqual(ipex_q, ref_q)
            self.assertEqual(ref_k, ipex_k)


if __name__ == "__main__":
    test = unittest.main()
