import torch
import math
import intel_extension_for_pytorch as ipex  # noqa

from torch.testing._internal.common_utils import TestCase
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="ipex build without xetla")
    def test_fsdp_atten_mask_alibi_stride(self, dtype=torch.float16):
        beam_width = 1
        num_heads = 14  # (/rank=8, 14)
        head_dim = 128
        q_len = 1
        kv_len = 1025  # 1152
        alibi_max_len = 2048  # for alignment restriction of Xe2DLoad

        beta = 1.0
        inv_norm_factor = 1.0 / math.sqrt(head_dim)

        print("CPU sdp ...")
        alibi = torch.randn(beam_width * num_heads, 1, kv_len).half()
        alibi_padded = torch.randn(beam_width * num_heads, 1, alibi_max_len).half()
        alibi_padded[:, :, 0:kv_len] = alibi
        # alibi.fill_(0)
        # alibi[1][:] = alibi[0][:]
        # print(alibi)
        query_layer = torch.randn(beam_width, q_len, num_heads, head_dim).half()
        key_layer = torch.randn(beam_width, kv_len, num_heads, head_dim).half()
        value_layer = torch.randn(beam_width, kv_len, num_heads, head_dim).half()

        print(key_layer.permute(0, 2, 3, 1).shape)
        print(key_layer.permute(0, 2, 3, 1).stride())
        gemm0_res = alibi.float().baddbmm(
            batch1=query_layer.float().permute(0, 2, 1, 3).view(-1, q_len, head_dim),
            batch2=key_layer.float().permute(0, 2, 3, 1).view(-1, head_dim, kv_len),
            beta=beta,
            alpha=inv_norm_factor,
        )

        attn_scores = gemm0_res.view(beam_width, num_heads, q_len, kv_len)
        # attn_mask = torch.zeros(beam_width, kv_len, dtype=torch.bool, device=torch.device('xpu'))
        # attn_weights = torch.masked_fill(attn_scores, attn_mask, torch.finfo(attn_scores.dtype).min)
        attn_weights = attn_scores
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_probs_reshaped = attn_probs.view(beam_width * num_heads, q_len, kv_len)
        context_layer = torch.bmm(
            attn_probs_reshaped,
            value_layer.float()
            .permute(0, 2, 1, 3)
            .view(-1, kv_len, head_dim)
            .view(beam_width * num_heads, kv_len, head_dim),
        )
        # print(context_layer.cpu().view(-1, q_len, head_dim))

        print("XPU sdp ...")
        alibi_sdp = alibi_padded.view(beam_width, num_heads, 1, alibi_max_len)
        print(alibi_sdp.shape)
        print(alibi_sdp.stride())
        # alibi_sdp = alibi.view(beam_width, num_heads, 1, kv_len)
        attn_mask = None
        head_mask = None
        alpha = inv_norm_factor
        beta = 1.0
        dropout = 0.0
        is_causal = False
        context_layer_sdp = torch.xpu.IpexSDP(
            query_layer.to("xpu").permute(0, 2, 1, 3),
            key_layer.to("xpu").permute(0, 2, 1, 3),
            value_layer.to("xpu").permute(0, 2, 1, 3),
            alibi_sdp.to("xpu"),
            attn_mask,
            head_mask,
            alpha,
            beta,
            dropout,
            is_causal,
        )
        print(context_layer_sdp.shape)
        print(context_layer_sdp.cpu().float())
        print(context_layer.shape)
        print(context_layer.cpu())
        self.assertEqual(
            context_layer.unsqueeze(0).cpu(),
            context_layer_sdp.cpu().float(),
            atol=1e-3,
            rtol=1e-4,
        )
