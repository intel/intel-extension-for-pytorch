import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch as ipex  # noqa
import pytest


def ipex_addmm(m, n, k, dtype):
    a = torch.rand(m, k).type(dtype).xpu()
    b = torch.rand(k, n).type(dtype).xpu()
    s = torch.rand(m, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    s_cpu = s.cpu().float()
    c = torch.addmm(s, a, b, alpha=0.2, beta=0.5)
    c_cpu = torch.addmm(s_cpu, a_cpu, b_cpu, alpha=0.2, beta=0.5)
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print("ipex_addmm:", (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_common(m, n, k, dtype):
    a = torch.rand(2, m, k).type(dtype).xpu()
    b = torch.rand(k, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    c = torch.ops.torch_ipex.mm_common(a, b)
    c_cpu = a_cpu @ b_cpu
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print("ipex_mm_common:", (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_resadd(m, n, k, dtype):
    a = torch.rand(2, m, k).type(dtype).xpu()
    b = torch.rand(k, n).type(dtype).xpu()
    res = torch.rand(2, m, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    res_cpu = res.cpu().float()
    c = torch.ops.torch_ipex.mm_resadd(a, b, res, 5.5)
    c_cpu = a_cpu @ b_cpu + 5.5 * res_cpu
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print("ipex_mm_resadd:", (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_resadd_resadd(m, n, k, dtype):
    a = torch.rand(2, m, k).type(dtype).xpu()
    b = torch.rand(k, n).type(dtype).xpu()
    res0 = torch.rand(2, m, n).type(dtype).xpu()
    res1 = torch.rand(2, m, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    res0_cpu = res0.cpu().float()
    res1_cpu = res1.cpu().float()
    c = torch.ops.torch_ipex.mm_resadd_resadd(a, b, res0, 5.5, res1, 4.5)
    c_cpu = a_cpu @ b_cpu + 5.5 * res0_cpu + 4.5 * res1_cpu
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print("ipex_mm_resadd_resadd:", (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_bias(m, n, k, dtype):
    a = torch.rand(2, m, k).type(dtype).xpu()
    b = torch.rand(k, n).type(dtype).xpu()
    bias = torch.rand(n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    bias_cpu = bias.cpu().float()
    c = torch.ops.torch_ipex.mm_bias(a, b, bias, 3.3)
    c_cpu = a_cpu @ b_cpu + 3.3 * bias_cpu
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print("ipex_mm_bias:", (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_bias_resadd(m, n, k, dtype):
    a = torch.rand(2, m, k).type(dtype).xpu()
    b = torch.rand(k, n).type(dtype).xpu()
    bias = torch.rand(n).type(dtype).xpu()
    res = torch.rand(2, m, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    bias_cpu = bias.cpu().float()
    res_cpu = res.cpu().float()
    c = torch.ops.torch_ipex.mm_bias_resadd(a, b, bias, 4.4, res, 6.6)
    c_cpu = a_cpu @ b_cpu + 4.4 * bias_cpu + 6.6 * res_cpu
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print("ipex_mm_bias_resadd:", (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_bias_resadd_resadd(m, n, k, dtype):
    a = torch.rand(2, m, k).type(dtype).xpu()
    b = torch.rand(k, n).type(dtype).xpu()
    bias = torch.rand(n).type(dtype).xpu()
    res0 = torch.rand(2, m, n).type(dtype).xpu()
    res1 = torch.rand(2, m, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    bias_cpu = bias.cpu().float()
    res0_cpu = res0.cpu().float()
    res1_cpu = res1.cpu().float()
    c = torch.ops.torch_ipex.mm_bias_resadd_resadd(
        a, b, bias, 5.5, res0, 6.6, res1, 7.7
    )
    c_cpu = a_cpu @ b_cpu + 5.5 * bias_cpu + 6.6 * res0_cpu + 7.7 * res1_cpu
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print("ipex_mm_bias_resadd_resadd:", (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_resmul(m, n, k, dtype):
    a = torch.rand(2, m, k).type(dtype).xpu()
    b = torch.rand(k, n).type(dtype).xpu()
    res = torch.rand(2, m, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    res_cpu = res.cpu().float()
    c = torch.ops.torch_ipex.mm_resmul(a, b, res)
    c_cpu = (a_cpu @ b_cpu) * res_cpu
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print("ipex_mm_resmul:", (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_silu(m, n, k, dtype):
    a = torch.rand(2, m, k).type(dtype).xpu()
    b = torch.rand(k, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    c = torch.ops.torch_ipex.mm_silu(a, b)
    c_cpu = F.silu(a_cpu @ b_cpu)
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print("ipex_mm_silu:", (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_relu(m, n, k, dtype):
    a = torch.rand(2, m, k).type(dtype).xpu()
    b = torch.rand(k, n).type(dtype).xpu()
    bias = torch.rand(n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    c = torch.ops.torch_ipex.matmul_relu(a, b, bias, 1.0)
    c_cpu = F.relu(a_cpu @ b_cpu)
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print("ipex_mm_relu:", (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_gelu(m, n, k, dtype):
    a = torch.rand(2, m, k).type(dtype).xpu()
    b = torch.rand(k, n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    c = torch.ops.torch_ipex.matmul_gelu(a, b, None, 1.0, "tanh")
    c_cpu = F.gelu(a_cpu @ b_cpu)
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print("ipex_mm_gelu:", (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_bias_gelu(m, n, k, dtype):
    a = torch.rand(2, m, k).type(dtype).xpu()
    b = torch.rand(k, n).type(dtype).xpu()
    bias = torch.rand(n).type(dtype).xpu()
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    bias_cpu = bias.cpu().float()
    c = torch.ops.torch_ipex.matmul_gelu(a, b, bias, 6.6, "tanh")
    c_cpu = F.gelu(a_cpu @ b_cpu + 6.6 * bias_cpu)
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print("ipex_mm_bias_gelu:", (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_qkv(m, n, k, dtype, with_bias):
    a = torch.rand(2, m, k).type(dtype).xpu()
    wq = torch.rand(k, n).type(dtype).xpu()
    wk = torch.rand(k, n).type(dtype).xpu()
    wv = torch.rand(k, n).type(dtype).xpu()
    wqkv = torch.stack([wq, wk, wv]).contiguous().xpu()
    bias = torch.rand(3, n).type(dtype).xpu()

    a_cpu = a.cpu().float()
    wq_cpu = wq.cpu().float()
    wk_cpu = wk.cpu().float()
    wv_cpu = wv.cpu().float()
    # wqkv_cpu = wqkv.cpu().float()
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
    print("ipex_mm_qkv:", (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


def ipex_mm_qkv_group(m, n, k, dtype, with_bias):
    bs = 1
    a = torch.rand(bs, m, k).type(dtype).xpu()
    num_head = 64
    num_kv_head = 8
    head_dim = n // num_head
    group = num_head // num_kv_head + 2
    wq = torch.rand(k, num_kv_head, group - 2, head_dim).type(dtype).xpu()
    wk = torch.rand(k, num_kv_head, 1, head_dim).type(dtype).xpu()
    wv = torch.rand(k, num_kv_head, 1, head_dim).type(dtype).xpu()
    # -> [num_head//num_kv_head + 2, hidden_size, num_kv_head, head_dim]
    wqkv = torch.concat([wq, wk, wv], dim=2).permute(2, 0, 1, 3).contiguous().xpu()
    bq = torch.rand(num_kv_head, group - 2, head_dim).type(dtype).xpu()
    bk = torch.rand(num_kv_head, 1, head_dim).type(dtype).xpu()
    bv = torch.rand(num_kv_head, 1, head_dim).type(dtype).xpu()
    bqkv = (
        torch.concat([bq, bk, bv], dim=1)
        .permute(1, 0, 2)
        .flatten(1, 2)
        .contiguous()
        .xpu()
    )

    a_cpu = a.cpu().float()
    wq_cpu = wq.reshape(k, num_head * head_dim).cpu().float()
    wk_cpu = wk.reshape(k, num_kv_head * head_dim).cpu().float()
    wv_cpu = wv.reshape(k, num_kv_head * head_dim).cpu().float()
    bq_cpu = bq.flatten().cpu().float()
    bk_cpu = bk.flatten().cpu().float()
    bv_cpu = bv.flatten().cpu().float()

    c0 = torch.empty((group - 2, bs, m, num_kv_head * head_dim), dtype=dtype).xpu()
    c1 = torch.empty((bs, m, num_kv_head * head_dim), dtype=dtype).xpu()
    c2 = torch.empty((bs, m, num_kv_head * head_dim), dtype=dtype).xpu()
    wqkv = wqkv.view(-1, k, num_kv_head, head_dim).reshape(
        -1, k, num_kv_head * head_dim
    )
    if with_bias:
        # for i in range(num_head // num_kv_head):
        #    c0[i] = a @ wqkv[i] + bq[i]
        # c1 = a @ wqkv[i + 1] + bq[i+1]
        # c2 = a @ wqkv[i + 2] + bv[i+2]
        torch.ops.torch_ipex.mm_qkv_group_out(a, wqkv, bqkv, c0, c1, c2)

        c0_cpu = a_cpu @ wq_cpu + bq_cpu
        c1_cpu = a_cpu @ wk_cpu + bk_cpu
        c2_cpu = a_cpu @ wv_cpu + bv_cpu
    else:
        torch.ops.torch_ipex.mm_qkv_group_out(a, wqkv, None, c0, c1, c2)

        c0_cpu = a_cpu @ wq_cpu
        c1_cpu = a_cpu @ wk_cpu
        c2_cpu = a_cpu @ wv_cpu

    # from: [num_head//num_kv_head + 2, m, num_kv_head, head_dim]
    # to: [bs, m, num_kv_head * num_head//num_kv_head * head_dim]
    c0 = (
        c0.view(-1, bs, m, num_kv_head, head_dim)
        .permute(1, 2, 3, 0, 4)
        .reshape(bs, m, -1)
    )
    c1 = c1.reshape(bs, m, -1)
    c2 = c2.reshape(bs, m, -1)

    c = torch.concat([c0, c1, c2], dim=-1)
    c_cpu = torch.concat([c0_cpu, c1_cpu, c2_cpu], dim=-1)
    c_, c_cpu_ = c.cpu().float(), c_cpu
    print("ipex_mm_qkv_group:", (c_ - c_cpu_).abs().max().item())
    return c_, c_cpu_


class TestNNMethod(TestCase):
    @pytest.mark.skipif(
        (not torch.xpu.has_xetla()) or (not ipex._C._has_2d_block_array(0)),
        reason="ipex build without xetla or is atsm",
    )
    def test_gemm_xetla(self):
        shapes = [
            # m, n, k
            [3, 4096, 4096],
            [3, 4096, 16384],
            [3, 16384, 4096],
            [3, 32000, 4096],
            [1008, 8200, 512],
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
            out, ref = ipex_mm_bias_resadd_resadd(
                shape[0], shape[1], shape[2], torch.half
            )
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)

            out, ref = ipex_mm_resmul(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_silu(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_relu(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_gelu(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_bias_gelu(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)

            out, ref = ipex_mm_qkv(shape[0], shape[1], shape[2], torch.half, True)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_qkv(shape[0], shape[1], shape[2], torch.half, False)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.skipif(
        (not torch.xpu.has_xetla()) or (not ipex._C._has_2d_block_array(0)),
        reason="ipex build without xetla or is atsm",
    )
    def test_gemm_xetla_group(self):
        shapes = [
            # m, n, k
            [4, 2048, 8192],
            [1, 2048, 8192],
            [3, 4096, 16384],
            [3, 16384, 4096],
            [3, 32000, 4096],
            [1008, 8200, 512],
        ]
        for shape in shapes:
            print(shape)
            out, ref = ipex_mm_qkv_group(
                shape[0], shape[1], shape[2], torch.half, False
            )
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
            out, ref = ipex_mm_qkv_group(shape[0], shape[1], shape[2], torch.half, True)
            self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.skipif(
        (not torch.xpu.has_xetla()) or (not ipex._C._has_2d_block_array(0)),
        reason="ipex build without xetla or is atsm",
    )
    def test_gemm_xetla_onednn(self):
        shapes = [
            # m, n, k
            [3, 4096, 4096],
            [3, 4096, 16384],
            [3, 16384, 4096],
            [3, 32000, 4096],
            [1008, 8200, 512],
        ]
        for shape in shapes:
            print(shape)

            # test gemm with onednn compute engine
            with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.ONEDNN):
                out, ref = ipex_addmm(shape[0], shape[1], shape[2], torch.half)
                self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
                out, ref = ipex_mm_common(shape[0], shape[1], shape[2], torch.half)
                self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
                out, ref = ipex_mm_resadd(shape[0], shape[1], shape[2], torch.half)
                self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
                out, ref = ipex_mm_resadd_resadd(
                    shape[0], shape[1], shape[2], torch.half
                )
                self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
                out, ref = ipex_mm_bias(shape[0], shape[1], shape[2], torch.half)
                self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
                out, ref = ipex_mm_bias_resadd(shape[0], shape[1], shape[2], torch.half)
                self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
                out, ref = ipex_mm_bias_resadd_resadd(
                    shape[0], shape[1], shape[2], torch.half
                )
                self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)

                out, ref = ipex_mm_resmul(shape[0], shape[1], shape[2], torch.half)
                self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
                out, ref = ipex_mm_silu(shape[0], shape[1], shape[2], torch.half)
                self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
                out, ref = ipex_mm_relu(shape[0], shape[1], shape[2], torch.half)
                self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
                out, ref = ipex_mm_gelu(shape[0], shape[1], shape[2], torch.half)
                self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
                out, ref = ipex_mm_bias_gelu(shape[0], shape[1], shape[2], torch.half)
                self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)

                out, ref = ipex_mm_qkv(shape[0], shape[1], shape[2], torch.half, True)
                self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
                out, ref = ipex_mm_qkv(shape[0], shape[1], shape[2], torch.half, False)
                self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)
