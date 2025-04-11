import torch
import intel_extension_for_pytorch as ipex  # noqa
import math
import pytest

from torch.testing._internal.common_utils import TestCase


def naive_sdp(query, key, value, attention_mask, head_mask, alibi, alpha):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    attn_weights *= alpha

    if attention_mask is not None:
        attn_weights += attention_mask
        # the attn_weights should anyway bigger than dtype.min, I wonder if this is necessary
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )
    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float
    ).to(query.dtype)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


dtype = dtype = torch.float16

checking_atol = 1e-3
checking_rtol = 1e-3

f = 1
t_in = 1919
t_out = 111
t = t_in + t_out
t_max = 2176
beam = 4
bs = 1
b = bs * beam
n = 1
h = 256

alpha = 1.0 / math.sqrt(h)
beta = 1.0
dropout = 0.0
is_causal = False
seq_last = True
alibi = None
attn_mask = None
attn_mask_padded = None
head_mask = None


print("alpha:", end=" ")
print(alpha)


class TestTorchMethod(TestCase):
    # Layout: SequenceLast
    # not support on DG2 yet
    @pytest.mark.skipif(
        not torch.xpu.has_2d_block_array(), reason="ipex build without xetla"
    )
    def test_fsdp_index_select(self, dtype=torch.float16):
        q = torch.randn([f, b, n, h], dtype=dtype, device=torch.device("xpu"))
        k_in_proj = torch.randn(
            [t_in, bs, n, h], dtype=dtype, device=torch.device("xpu")
        )
        v_in_proj = torch.randn(
            [t_in, bs, n, h], dtype=dtype, device=torch.device("xpu")
        )
        k = torch.randn([t, b, n, h], dtype=dtype, device=torch.device("xpu"))
        v = torch.randn([t, b, n, h], dtype=dtype, device=torch.device("xpu"))
        k_cache = torch.randn([t_max, b, n, h], dtype=dtype, device=torch.device("xpu"))
        v_cache = torch.randn([t_max, b, n, h], dtype=dtype, device=torch.device("xpu"))
        attn_mask = torch.randn([b, n, f, t], dtype=dtype, device=torch.device("xpu"))
        attn_mask_padded = torch.zeros(
            [b, n, f, t_max], dtype=dtype, device=torch.device("xpu")
        )
        index = torch.randint(
            0, beam, [t_out, bs, beam], dtype=torch.int, device=torch.device("xpu")
        )
        # print("index", index)

        # Reference init
        # 1. Init k, v 1st half, input prompt projection
        # broadcast bs -> bs*beam
        # print("before bc k", k[0:t_in,:,:,:])
        k_ = k[0:t_in, :, :, :].view(t_in, bs, beam, n, h)
        k_.copy_(k_in_proj.view(t_in, bs, 1, n, h))
        v_ = v[0:t_in, :, :, :].view(t_in, bs, beam, n, h)
        v_.copy_(v_in_proj.view(t_in, bs, 1, n, h))
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
                k[t_in + i][j].copy_(k_history[i][int(bs_)][beam_choice])
                v[t_in + i][j].copy_(v_history[i][int(bs_)][beam_choice])

        attn_mask_padded[:, :, :, 0:t] = attn_mask

        # print("q", q)
        # print("k_in_proj", k_in_proj)
        # print("k_cache", k_cache[0:t_out,:,:,:])
        # print("v_in_proj", v_in_proj)
        # print("v_cache", v_cache[0:t_out,:,:,:])
        # print("k", k)
        # print("v", v)

        naive, _ = naive_sdp(
            q.permute(1, 2, 0, 3),
            k.permute(1, 2, 0, 3),
            v.permute(1, 2, 0, 3),
            attn_mask,
            None,
            None,
            alpha,
        )

        ref = torch.xpu.IpexSDP(
            q.permute(1, 2, 0, 3),
            k.permute(1, 2, 0, 3),
            v.permute(1, 2, 0, 3),
            alibi,
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
            q.permute(1, 2, 0, 3),
            k_in_proj.permute(1, 2, 0, 3),
            v_in_proj.permute(1, 2, 0, 3),
            k_cache[0:t_out, :, :, :].permute(1, 2, 0, 3),
            v_cache[0:t_out, :, :, :].permute(1, 2, 0, 3),
            index,
            alibi,
            attn_mask_padded,
            head_mask,
            t_out,
            alpha,
            beta,
            dropout,
            is_causal,
        )

        print(
            "sdp vs sdp_index: ", torch.max(torch.abs(res.cpu() - naive.cpu())).item()
        )
        print("sdp vs naive: ", torch.max(torch.abs(ref.cpu() - naive.cpu())).item())
        self.assertEqual(ref.cpu(), naive.cpu(), atol=1e-2, rtol=1e-3)
        # self.assertEqual(res.cpu(), naive.cpu(), atol=1e-2, rtol=1e-3)
