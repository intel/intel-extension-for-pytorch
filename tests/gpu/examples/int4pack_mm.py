import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

checking_atol = 1e-2
checking_rtol = 1e-2


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="ipex build without xetla")
    def test_int4pack_with_zero_x_scale(self):
        N_L = [
            16,
            32,
            48,
            64,
            96,
            128,
            160,
            256,
            320,
            400,
            4096,
            11008,
            12288,
            32000,
            10240,
            8192,
            28672,
        ]
        K_L = [32, 64, 128, 256, 4096, 11008, 12288, 8192, 28672]
        if not (torch.xpu.has_2d_block_array() or torch.xpu.has_xmx()):
            pass
        else:
            for N in N_L:
                for K in K_L:
                    w = torch.randint(0, 16, (N, K), dtype=torch.int)
                    q_int4 = torch.ops.aten._convert_weight_to_int4pack(w.to("xpu"), 2)
                    for data_type in [torch.bfloat16]:
                        scales_and_zeros = torch.randn(
                            (int(K / 32), N, 2), dtype=data_type
                        )
                        s_bf16 = (
                            scales_and_zeros[:, :, 0]
                            .reshape(int(K / 32), N, 1)
                            .transpose(0, 1)
                        )
                        z_bf16 = (
                            scales_and_zeros[:, :, 1]
                            .reshape(int(K / 32), N, 1)
                            .transpose(0, 1)
                        )
                        w_cpu = (
                            w.reshape((N, int(K / 32), 32)) * s_bf16
                            + z_bf16
                            - s_bf16 * 8
                        )
                        for M in range(1, 17):
                            x = torch.randn((M, K), dtype=data_type)
                            out_xpu = torch.ops.aten._weight_int4pack_mm(
                                x.to("xpu"),
                                q_int4.to("xpu"),
                                32,
                                scales_and_zeros.to("xpu"),
                            )
                            out_cpu = torch.nn.functional.linear(
                                x, w_cpu.reshape(N, K).to(data_type)
                            )
                            self.assertEqual(
                                out_cpu,
                                out_xpu.cpu(),
                                atol=checking_atol,
                                rtol=checking_rtol,
                            )
