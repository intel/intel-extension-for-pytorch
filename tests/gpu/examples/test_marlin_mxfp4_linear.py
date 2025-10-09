import torch
import pytest
import numpy as np
import intel_extension_for_pytorch  # noqa

from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


checking_atol = 1e-2
checking_rtol = 1e-2


class TestMxfp4Linear(TestCase):
    def dequantize(self, qweight, scales, group_size):
        k = qweight.shape[0] * 2
        n = qweight.shape[1]
        # use pre-shuffle
        unpack_idx = np.array([0, 1])
        data = qweight[[i // 2 for i in range(k)], :]
        shift = (
            torch.tensor(
                unpack_idx[[i % 2 for i in range(k)]], dtype=torch.int32, device="xpu"
            )[:, None].expand([-1, n])
            * 4
        )
        dst_data = (data >> shift) & 0xF

        table = torch.tensor(
            [
                0b0000000000000000,
                0b0011111100000000,
                0b0011111110000000,
                0b0011111111000000,
                0b0100000000000000,
                0b0100000001000000,
                0b0100000010000000,
                0b0100000011000000,
                0b1000000000000000,
                0b1011111100000000,
                0b1011111110000000,
                0b1011111111000000,
                0b1100000000000000,
                0b1100000001000000,
                0b1100000010000000,
                0b1100000011000000,
            ],
            dtype=torch.int32,
            device="xpu",
        )
        # table = torch.tensor(
        #     [+0.0,
        #      0.5,
        #      1.0,
        #      1.5,
        #      2.0,
        #      3.0,
        #      4.0,
        #      6.0,
        #      -0.0,
        #      -0.5,
        #      -1.0,
        #      -1.5,
        #      -2.0,
        #      -3.0,
        #      -4.0,
        #      -6.0,
        #     ], dtype=torch.bfloat16, device="xpu")
        dst_data = table[dst_data].to(torch.uint16).view(torch.bfloat16)
        expand_scales = scales[[i // group_size for i in range(k)], :]
        dst_scale = (
            (expand_scales.to(torch.int32) << 7).to(torch.uint16).view(torch.bfloat16)
        )
        weight_bf16 = dst_data * dst_scale

        return weight_bf16

    def shuffle_weight(self, qweight):
        k = qweight.shape[0] * 8
        n = qweight.shape[1]
        shuffled_idx = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        data = qweight[[i // 8 for i in range(k)], :]
        shift = (
            torch.tensor(
                shuffled_idx[[i % 8 for i in range(k)]], dtype=torch.int32, device="xpu"
            )[:, None].expand([-1, n])
            * 4
        )
        dst_data = (data >> shift) & 0xF
        # compressed back to int32
        pack_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        shift = (
            torch.tensor(
                pack_idx[[i % 8 for i in range(k)]], dtype=torch.int32, device="xpu"
            )[:, None].expand([-1, n])
            * 4
        )
        dst_data = dst_data << shift
        # print(dst_data.shape)
        shuffled_weight = torch.zeros([k // 8, n], dtype=torch.int32, device="xpu")
        for i in range(0, k, 8):
            tmp = dst_data[i, :]
            for j in range(i + 1, i + 8):
                tmp = torch.bitwise_or(tmp, dst_data[j, :])
            shuffled_weight[i // 8, :] = tmp
        # print(shuffled_weight.shape)
        return shuffled_weight

    @parametrize("per_channel", [False], lambda k: "per_channel" * k)
    @parametrize(
        "m,n,k", [(1, 4096, 4096), (8, 4096, 4096), (32, 4096, 4096), (1, 4096, 11008)]
    )
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_gemm_mxfp4(self, m, n, k, per_channel, dtype=torch.bfloat16):
        torch.manual_seed(0)
        input = torch.rand([m, k], device="xpu", dtype=dtype)
        input_torch = input.cpu()
        weight = torch.randint(0, 0xFF, [k // 2, n], dtype=torch.uint8, device="xpu")

        group_size = 32
        group_num = k // group_size

        scales = torch.randint(0, 0xEF, [group_num, n], dtype=torch.uint8, device="xpu")

        weight_bf16 = self.dequantize(weight, scales, group_size).cpu()

        weight = (
            weight.view(k // 8, 4, n)
            .permute(0, 2, 1)
            .reshape(k // 8, n * 4)
            .view(torch.int32)
        )
        weight = self.shuffle_weight(weight)
        out_xetla = torch.empty([m, n], dtype=torch.bfloat16, device="xpu")
        torch.ops.torch_ipex.mm_mxfp4_out_marlin(
            out_xetla, input, weight, scales, group_size
        )

        out_torch = torch.matmul(input_torch, weight_bf16)

        self.assertEqual(
            out_xetla.cpu().float(),
            out_torch.float(),
            atol=checking_atol,
            rtol=checking_rtol,
            equal_nan=True,
        )


instantiate_parametrized_tests(TestMxfp4Linear)

if __name__ == "__main__":
    run_tests()
