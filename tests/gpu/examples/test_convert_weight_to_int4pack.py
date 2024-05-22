import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

checking_atol = 1e-3
checking_rtol = 1e-3


class TestTorchMethod(TestCase):
    def test_convert_weight_to_int4pack(self):
        N_L = [16, 32, 48, 64, 96, 128, 160, 256, 320, 400, 4096, 11008, 12288]
        for N in N_L:
            weight_cpu = torch.randint(0, 16, (N, 32), dtype=torch.int)
            weight_xpu = weight_cpu.to("xpu")
            weight_int4_xpu = torch.ops.aten._convert_weight_to_int4pack(
                weight_xpu, 2
            ).reshape(32, int(N / 8))
            weight_int4_0 = weight_int4_xpu & 0x0000000F
            weight_int4_1 = (weight_int4_xpu & 0x000000F0) >> 4
            weight_int4_2 = (weight_int4_xpu & 0x00000F00) >> 8
            weight_int4_3 = (weight_int4_xpu & 0x0000F000) >> 12
            weight_int4_4 = (weight_int4_xpu & 0x000F0000) >> 16
            weight_int4_5 = (weight_int4_xpu & 0x00F00000) >> 20
            weight_int4_6 = (weight_int4_xpu & 0x0F000000) >> 24
            weight_int4_7 = (weight_int4_xpu >> 28) & 0x0000000F
            self.assertEqual(
                weight_cpu.t()[:, 0:N:8],
                weight_int4_0.cpu(),
                atol=checking_atol,
                rtol=checking_rtol,
            )
            self.assertEqual(
                weight_cpu.t()[:, 1:N:8],
                weight_int4_1.cpu(),
                atol=checking_atol,
                rtol=checking_rtol,
            )
            self.assertEqual(
                weight_cpu.t()[:, 2:N:8],
                weight_int4_2.cpu(),
                atol=checking_atol,
                rtol=checking_rtol,
            )
            self.assertEqual(
                weight_cpu.t()[:, 3:N:8],
                weight_int4_3.cpu(),
                atol=checking_atol,
                rtol=checking_rtol,
            )
            self.assertEqual(
                weight_cpu.t()[:, 4:N:8],
                weight_int4_4.cpu(),
                atol=checking_atol,
                rtol=checking_rtol,
            )
            self.assertEqual(
                weight_cpu.t()[:, 5:N:8],
                weight_int4_5.cpu(),
                atol=checking_atol,
                rtol=checking_rtol,
            )
            self.assertEqual(
                weight_cpu.t()[:, 6:N:8],
                weight_int4_6.cpu(),
                atol=checking_atol,
                rtol=checking_rtol,
            )
            self.assertEqual(
                weight_cpu.t()[:, 7:N:8],
                weight_int4_7.cpu(),
                atol=checking_atol,
                rtol=checking_rtol,
            )
