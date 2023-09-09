import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


def ipex_addmm(m, n, k, dtype):
    a = torch.randn(m, k).type(dtype).xpu()
    b = torch.randn(k, n).type(dtype).xpu()
    s = torch.randn(m, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    s_cpu = s.cpu().float()
    c = torch.addmm(s, a, b, alpha=0.2, beta=0.5)
    c_cpu = torch.addmm(s_cpu, a_cpu, b_cpu, alpha=0.2, beta=0.5)
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print('ipex_addmm:', (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_common(m, n, k, dtype):
    a = torch.randn(2, m, k).type(dtype).xpu()
    b = torch.randn(k, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    c = torch.ops.torch_ipex.mm_common(a, b)
    c_cpu = a_cpu @ b_cpu
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print('ipex_mm_common:', (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_resadd(m, n, k, dtype):
    a = torch.randn(2, m, k).type(dtype).xpu()
    b = torch.randn(k, n).type(dtype).xpu()
    res = torch.randn(2, m, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    res_cpu = res.cpu().float()
    c = torch.ops.torch_ipex.mm_resadd(a, b, res, 5.5)
    c_cpu = a_cpu @ b_cpu + 5.5 * res_cpu
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print('ipex_mm_resadd:', (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_resadd_resadd(m, n, k, dtype):
    a = torch.randn(2, m, k).type(dtype).xpu()
    b = torch.randn(k, n).type(dtype).xpu()
    res0 = torch.randn(2, m, n).type(dtype).xpu()
    res1 = torch.randn(2, m, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    res0_cpu = res0.cpu().float()
    res1_cpu = res1.cpu().float()
    c = torch.ops.torch_ipex.mm_resadd_resadd(a, b, res0, 5.5, res1, 4.5)
    c_cpu = a_cpu @ b_cpu + 5.5 * res0_cpu + 4.5 * res1_cpu
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print('ipex_mm_resadd_resadd:', (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_bias(m, n, k, dtype):
    a = torch.randn(2, m, k).type(dtype).xpu()
    b = torch.randn(k, n).type(dtype).xpu()
    bias = torch.randn(n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    bias_cpu = bias.cpu().float()
    c = torch.ops.torch_ipex.mm_bias(a, b, bias, 3.3)
    c_cpu = a_cpu @ b_cpu + 3.3 * bias_cpu
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print('ipex_mm_bias:', (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_bias_resadd(m, n, k, dtype):
    a = torch.randn(2, m, k).type(dtype).xpu()
    b = torch.randn(k, n).type(dtype).xpu()
    bias = torch.randn(n).type(dtype).xpu()
    res = torch.randn(2, m, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    bias_cpu = bias.cpu().float()
    res_cpu = res.cpu().float()
    c = torch.ops.torch_ipex.mm_bias_resadd(a, b, bias, 4.4, res, 6.6)
    c_cpu = a_cpu @ b_cpu + 4.4 * bias_cpu + 6.6 * res_cpu
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print('ipex_mm_bias_resadd:', (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_bias_resadd_resadd(m, n, k, dtype):
    a = torch.randn(2, m, k).type(dtype).xpu()
    b = torch.randn(k, n).type(dtype).xpu()
    bias = torch.randn(n).type(dtype).xpu()
    res0 = torch.randn(2, m, n).type(dtype).xpu()
    res1 = torch.randn(2, m, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    bias_cpu = bias.cpu().float()
    res0_cpu = res0.cpu().float()
    res1_cpu = res1.cpu().float()
    c = torch.ops.torch_ipex.mm_bias_resadd_resadd(a, b, bias, 5.5, res0, 6.6, res1, 7.7)
    c_cpu = a_cpu @ b_cpu + 5.5 * bias_cpu + 6.6 * res0_cpu + 7.7 * res1_cpu
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print('ipex_mm_bias_resadd_resadd:', (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_resmul(m, n, k, dtype):
    a = torch.randn(2, m, k).type(dtype).xpu()
    b = torch.randn(k, n).type(dtype).xpu()
    res = torch.randn(2, m, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    res_cpu = res.cpu().float()
    c = torch.ops.torch_ipex.mm_resmul(a, b, res)
    c_cpu = (a_cpu @ b_cpu) * res_cpu
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print('ipex_mm_resmul:', (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_silu(m, n, k, dtype):
    a = torch.randn(2, m, k).type(dtype).xpu()
    b = torch.randn(k, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    c = torch.ops.torch_ipex.mm_silu(a, b)
    c_cpu = F.silu(a_cpu @ b_cpu)
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print('ipex_mm_silu:', (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_gelu(m, n, k, dtype):
    a = torch.randn(2, m, k).type(dtype).xpu()
    b = torch.randn(k, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    c = torch.ops.torch_ipex.matmul_gelu(a, b, None, 1.0, 'tanh')
    c_cpu = F.gelu(a_cpu @ b_cpu)
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print('ipex_mm_gelu:', (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_bias_gelu(m, n, k, dtype):
    a = torch.randn(2, m, k).type(dtype).xpu()
    b = torch.randn(k, n).type(dtype).xpu()
    bias = torch.randn(n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    bias_cpu = bias.cpu().float()
    c = torch.ops.torch_ipex.matmul_gelu(a, b, bias, 6.6, 'tanh')
    c_cpu = F.gelu(a_cpu @ b_cpu + 6.6 * bias_cpu)
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print('ipex_mm_bias_gelu:', (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_qkv(m, n, k, dtype, with_bias):
    a = torch.randn(2, m, k).type(dtype).xpu()
    wq = torch.randn(k, n).type(dtype).xpu()
    wk = torch.randn(k, n).type(dtype).xpu()
    wv = torch.randn(k, n).type(dtype).xpu()
    wqkv = torch.stack([wq, wk, wv]).contiguous().xpu()
    bias = torch.randn(3, n).type(dtype).xpu()

    a_cpu = a.cpu().float()
    wq_cpu = wq.cpu().float()
    wk_cpu = wk.cpu().float()
    wv_cpu = wv.cpu().float()
    wqkv_cpu = wqkv.cpu().float()
    bias_cpu = bias.cpu().float()

    if with_bias:
        c0, c1, c2 = torch.ops.torch_ipex.mm_qkv(a, wqkv, bias)
        c0_cpu = a_cpu @ wq_cpu + bias_cpu[0]
        c1_cpu = a_cpu @ wk_cpu + bias_cpu[1]
        c2_cpu = a_cpu @ wv_cpu + bias_cpu[2]
    else:
        c0, c1, c2 = torch.ops.torch_ipex.mm_qkv(a, wqkv, None)
        c0_cpu = a_cpu @ wq_cpu
        c1_cpu = a_cpu @ wk_cpu
        c2_cpu = a_cpu @ wv_cpu

    c = torch.stack([c0, c1, c2])
    c_cpu = torch.stack([c0_cpu, c1_cpu, c2_cpu])
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print('ipex_mm_qkv:', (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


class TestNNMethod(TestCase):
    def test_gemm_xetla(self):
        shapes = [
            # m, n, k
            [3, 4096, 4096],
            [3, 4096, 16384],
            [3, 16384, 4096],
            [3, 32000, 4096]
        ]
        for shape in shapes:
            print(shape)
            out, ref = ipex_addmm(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)

            out, ref = ipex_mm_common(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_resadd(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_resadd_resadd(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_bias(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_bias_resadd(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_bias_resadd_resadd(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)

            out, ref = ipex_mm_resmul(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_silu(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_gelu(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_bias_gelu(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)

            out, ref = ipex_mm_qkv(shape[0], shape[1], shape[2], torch.half, True)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_qkv(shape[0], shape[1], shape[2], torch.half, False)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
