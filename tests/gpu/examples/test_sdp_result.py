import torch
from torch.nn import functional as F
import intel_extension_for_pytorch as ipex  # noqa

from torch.testing._internal.common_utils import TestCase
import pytest


class TestTorchMethod(TestCase):
    # not support on DG2 yet
    @pytest.mark.skipif(
        not torch.xpu.has_2d_block_array(), reason="ipex build without xetla"
    )
    def test_sdp_result(self, dtype=torch.float16):
        torch.manual_seed(42)

        q_cpu = torch.randn((4, 71, 32, 64))
        k_cpu = torch.randn((4, 1, 32, 64))
        v_cpu = torch.randn((4, 1, 32, 64))
        mask_cpu = torch.randn((4, 1, 32, 32))

        q_xpu = q_cpu.half().to("xpu")
        k_xpu = k_cpu.half().to("xpu")
        v_xpu = v_cpu.half().to("xpu")
        mask_xpu = mask_cpu.half().to("xpu")

        cpu_output = (
            F.scaled_dot_product_attention(
                q_cpu, k_cpu, v_cpu, mask_cpu, 0.0, is_causal=False
            )
            .to("cpu")
            .float()
        )

        xpu_math_output = (
            F.scaled_dot_product_attention(
                q_xpu, k_xpu, v_xpu, mask_xpu, 0.0, is_causal=False
            )
            .to("cpu")
            .float()
        )

        xpu_effi_output = (
            F.scaled_dot_product_attention(
                q_xpu,
                k_xpu.broadcast_to((4, 71, 32, 64)),
                v_xpu.broadcast_to((4, 71, 32, 64)),
                mask_xpu,
                0.0,
                is_causal=False,
            )
            .to("cpu")
            .float()
        )

        self.assertEqual(
            cpu_output,
            xpu_math_output,
            atol=1e-2,
            rtol=1e-4,
        )

        self.assertEqual(
            xpu_effi_output,
            xpu_math_output,
            atol=1e-2,
            rtol=1e-4,
        )
