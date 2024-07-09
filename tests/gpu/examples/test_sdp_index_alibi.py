import torch
import math
import intel_extension_for_pytorch as ipex  # noqa

from torch.testing._internal.common_utils import TestCase
import pytest

checking_atol = 1e-3
checking_rtol = 1e-3

f = 1
t_in = 1919
t_out = 111
t = t_in + t_out
t_max = 2176
beam = 4
bs = 64
b = bs * beam
n = 16
h = 256

alpha = 1.0 / math.sqrt(h)
beta = 1.0
dropout = 0.0
is_causal = False
seq_last = False
alibi = None
attn_mask = None
attn_mask_padded = None
alibi_max_len = 2048  # for alignment restriction of Xe2DLoad
head_mask = None


class TestTorchMethod(TestCase):
    # not support on DG2 yet
    @pytest.mark.skipif(
        not torch.xpu.has_2d_block_array(), reason="ipex build without xetla"
    )
    def test_fsdp_index_alibi(self, dtype=torch.float16):
        q = torch.randn([b, f, n, h], dtype=dtype, device=torch.device("xpu"))
        k_in_proj = torch.randn(
            [bs, t_in, n, h], dtype=dtype, device=torch.device("xpu")
        )
        v_in_proj = torch.randn(
            [bs, t_in, n, h], dtype=dtype, device=torch.device("xpu")
        )
        k = torch.randn([b, t, n, h], dtype=dtype, device=torch.device("xpu"))
        v = torch.randn([b, t, n, h], dtype=dtype, device=torch.device("xpu"))
        k_cache = torch.randn([t_max, b, n, h], dtype=dtype, device=torch.device("xpu"))
        v_cache = torch.randn([t_max, b, n, h], dtype=dtype, device=torch.device("xpu"))
        attn_mask = torch.randn([b, 1, f, t], dtype=dtype, device=torch.device("xpu"))
        attn_mask_padded = torch.zeros(
            [b, 1, f, t_max], dtype=dtype, device=torch.device("xpu")
        )
        index = torch.randint(
            0, beam, [t_out, bs, beam], dtype=torch.int, device=torch.device("xpu")
        )

        # Reference init
        # 1. Init k, v 1st half, input prompt projection
        # broadcast bs -> bs*beam
        # print("before bc k", k[0:t_in,:,:,:])
        k_ = k[:, 0:t_in, :, :].view(bs, beam, t_in, n, h)
        k_.copy_(k_in_proj.view(bs, 1, t_in, n, h))
        v_ = v[:, 0:t_in, :, :].view(bs, beam, t_in, n, h)
        v_.copy_(v_in_proj.view(bs, 1, t_in, n, h))
        # print("after bc k", k[0:t_in,:,:,:])

        # 2. Init k, v 2nd half, inference output tokens projection till last timestep
        # index select according to beam record index
        k_history = k_cache[0:t_out, :, :, :].view(t_out, bs, beam, n, h)
        v_history = v_cache[0:t_out, :, :, :].view(t_out, bs, beam, n, h)
        index_ = index.view(t_out, b)
        for i in range(t_out):
            for j in range(b):
                bs_ = j / beam
                beam_choice = index_[i, j]
                k[j][t_in + i].copy_(k_history[i][int(bs_)][beam_choice])
                v[j][t_in + i].copy_(v_history[i][int(bs_)][beam_choice])

        # 3. Init Alibi
        # alibi = torch.randn(b * n, 1, t).half()
        alibi = torch.randn(b * n, 1, t).half()
        alibi_padded = torch.randn(b * n, 1, alibi_max_len).half()
        alibi_padded[:, :, 0:t] = alibi
        # alibi.fill_(0)
        # alibi[1][:] = alibi[0][:]
        # print(alibi)

        # 4. Init Attn mask
        attn_mask_padded = None  # TODO: no reference path?

        print("CPU sdp ...")
        print(k.cpu().float().permute(0, 2, 3, 1).shape)
        gemm0_res = alibi.float().baddbmm(
            batch1=q.cpu().float().permute(0, 2, 1, 3).contiguous().view(b * n, f, h),
            batch2=k.cpu().float().permute(0, 2, 3, 1).contiguous().view(b * n, h, t),
            beta=beta,
            alpha=alpha,
        )

        attn_scores = gemm0_res.view(b, n, f, t)
        # attn_mask = torch.zeros(b, t, dtype=torch.bool, device=torch.device('xpu'))
        # attn_weights = torch.masked_fill(attn_scores, attn_mask, torch.finfo(attn_scores.dtype).min)
        attn_weights = attn_scores
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_probs_reshaped = attn_probs.view(b * n, f, t)
        ref_cpu = torch.bmm(
            attn_probs_reshaped,
            v.cpu().float().permute(0, 2, 1, 3).contiguous().view(-1, t, h),
        ).view(b, n, f, h)
        # print(context_layer.cpu().view(-1, f, h))

        print("XPU sdp ...")
        alibi_sdp = alibi_padded.xpu().view(b, n, 1, alibi_max_len)
        print(alibi_sdp.shape)
        print(alibi_sdp.stride())
        # alibi_sdp = None

        ref = torch.xpu.IpexSDP(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            alibi_sdp,
            attn_mask_padded,
            head_mask,
            alpha,
            beta,
            dropout,
            is_causal,
            seq_last,
        )
        ref_no_alibi = torch.xpu.IpexSDP(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            None,
            attn_mask_padded,
            head_mask,
            alpha,
            beta,
            dropout,
            is_causal,
            seq_last,
        )
        # sdp index fusion op is on SequenceLast by default
        res = torch.xpu.IpexSDP_Index(
            q.permute(0, 2, 1, 3),
            k_in_proj.permute(0, 2, 1, 3),
            v_in_proj.permute(0, 2, 1, 3),
            k_cache[0:t_out, :, :, :].permute(1, 2, 0, 3),
            v_cache[0:t_out, :, :, :].permute(1, 2, 0, 3),
            index,
            alibi_sdp,
            attn_mask_padded,
            head_mask,
            t_out,
            alpha,
            beta,
            dropout,
            is_causal,
        )
        # print(context_layer_sdp.cpu().view(-1, f, h))
        print("sdp vs sdp idx:", (ref.cpu() - res.cpu()).abs().max().item())
        print("cpu vs sdp idx:", (ref_cpu - res.cpu()).abs().max().item())
        print("cpu vs sdp:", (ref_cpu - ref.cpu()).abs().max().item())
        print(
            "sdp no alibi vs sdp idx:",
            (ref_no_alibi.cpu() - res.cpu()).abs().max().item(),
        )
        self.assertEqual(ref_cpu, ref.cpu().float(), atol=1e-2, rtol=1e-3)
        self.assertEqual(ref_cpu, res.cpu().float(), atol=1e-2, rtol=1e-3)
