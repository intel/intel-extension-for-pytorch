import torch
import intel_extension_for_pytorch as ipex  # noqa
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

    @pytest.mark.skipif(
        ipex._C._has_2d_block_array(0),
        reason="Only for naive sdp with half datatype on ATS-M",
    )
    def test_sdp_math_half(self, dtype=torch.float16):
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
