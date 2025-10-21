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


class TestInt4Linear(TestCase):
    def dequantize(self, qweight, scales, qzeros, group_size):
        k = qweight.shape[0] * 8
        n = qweight.shape[1]
        # use pre-shuffle
        # unpack_idx = np.array([0, 2, 4, 6, 1, 3, 5, 7])
        unpack_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        data = qweight[[i // 8 for i in range(k)], :]
        shift = (
            torch.tensor(
                unpack_idx[[i % 8 for i in range(k)]], dtype=torch.int32, device="xpu"
            )[:, None].expand([-1, n])
            * 4
        )
        dst_data = (data >> shift) & 0xF
        expand_scales = scales[[i // group_size for i in range(k)], :]
        weight_fp16 = (dst_data - 8) * expand_scales

        return weight_fp16

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
        "m,n,k",
        [
            (1, 4096, 4096),
            (4, 4096, 4096),
            (32, 4096, 4096),
            (128, 4096, 4096),
            (1024, 4096, 4096),
            (1, 4096, 11008),
        ],
    )
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_gemm_int4(self, m, n, k, per_channel, dtype=torch.float16):
        torch.manual_seed(0)
        input = torch.rand([m, k], device="xpu", dtype=dtype)
        input_torch = input.cpu().float()
        weight = (torch.randint(0, 11111111, [k // 8, n], device="xpu")).to(
            torch.int32
        ) * (-1)

        group_size = min(128, k)
        if per_channel:
            group_size = k
        group_num = int(k / group_size)

        scales = torch.rand([group_num, n], device="xpu", dtype=torch.float16)
        zero_points = torch.randint(0, 11111, [group_num, n // 8], device="xpu").to(
            torch.int32
        )

        weight_fp16 = (
            self.dequantize(weight, scales, zero_points, group_size).cpu().float()
        )

        weight = self.shuffle_weight(weight)
        out_xetla = torch.zeros([m, n], dtype=torch.float16, device="xpu")
        torch.ops.torch_ipex.mm_int4_out_marlin(
            out_xetla, input, weight, scales, None, group_size
        )
        out_torch = torch.matmul(input_torch, weight_fp16)

        self.assertEqual(
            out_xetla.cpu().float(),
            out_torch.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )


instantiate_parametrized_tests(TestInt4Linear)

if __name__ == "__main__":
    run_tests()
