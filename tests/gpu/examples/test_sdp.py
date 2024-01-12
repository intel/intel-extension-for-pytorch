import torch
import intel_extension_for_pytorch  # noqa
import torch.nn.functional as F

from torch.testing._internal.common_utils import TestCase
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_sdp_mem_effi_half(self, dtype=torch.float16):
        head_dim = 256
        seq_lenth = 1
        k_seq_lenth = 33
        v_seq_lenth = 33
        query = torch.rand(1, 16, seq_lenth, head_dim, dtype=dtype)
        key = torch.rand(1, 16, k_seq_lenth, head_dim, dtype=dtype)
        value = torch.rand(1, 16, v_seq_lenth, head_dim, dtype=dtype)

        out_cpu = F.scaled_dot_product_attention(
            query.float(), key.float(), value.float()
        )
        out_xpu = F.scaled_dot_product_attention(query.xpu(), key.xpu(), value.xpu())

        self.assertEqual(out_cpu, out_xpu.cpu().float(), atol=1e-3, rtol=1e-3)

    def test_sdp_broadcast(self, dtype=torch.float16):
        query = torch.rand(8, 8, 77, 64, dtype=dtype)
        key = torch.rand(8, 8, 77, 64, dtype=dtype)
        value = torch.rand(8, 8, 77, 64, dtype=dtype)

        bias = torch.rand(1, 1, 77, 77, dtype=dtype)

        out_cpu = F.scaled_dot_product_attention(
            query.float(), key.float(), value.float(), bias.float()
        )
        out_xpu = F.scaled_dot_product_attention(
            query.xpu(), key.xpu(), value.xpu(), bias.xpu()
        )

        self.assertEqual(out_cpu, out_xpu.cpu().float(), atol=1e-3, rtol=1e-3)
