import torch
import intel_extension_for_pytorch  # noqa
import pytest

from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


checking_atol = 1e-2
checking_rtol = 1e-2


class TestInt4Dequantize(TestCase):

    def unpack_weight(self, qweight, scales, qzeros, q_config):
        group_size = q_config["group_size"]
        bits = q_config["bits"]
        s32_bits = 32

        assert bits == 4
        # Int32 can store 8 * 4bits data. This is the offset for each data.
        wf = (
            torch.tensor(list(range(0, s32_bits, bits)), dtype=torch.int32)
            .unsqueeze(0)
            .to("xpu")
        )
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
        ).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)

        zeros = zeros + 1
        zeros = zeros.reshape(scales.shape)

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)
        ).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(weight, (2**bits) - 1, out=weight)

        return weight, scales, zeros

    def dequantize(self, qweight, scales, qzeros, group_size):
        q_config = {"group_size": group_size, "bits": 4}
        weight, gptq_scales, gptq_zeros = self.unpack_weight(
            qweight, scales, qzeros, q_config
        )
        gptq_zeros = (torch.ones_like(gptq_zeros) * 8).to("xpu")  # TODO: hard code zp
        if len(weight.shape) > 2:
            weight = weight.reshape(-1, weight.shape[-1])
        infeatures = weight.shape[0]
        g_idx = torch.tensor(
            [i // q_config["group_size"] for i in range(infeatures)], dtype=torch.int32
        )
        scale_zeros = gptq_zeros * gptq_scales
        weight = gptq_scales[g_idx.long()] * weight - scale_zeros[g_idx.long()]
        weight = weight.to(torch.float16)
        return weight

    @staticmethod
    def rand_int4(size, dtype=torch.int32, device="xpu"):
        rand = torch.randint(-128, 128, [size // 2], device=device).to(torch.int8)
        return rand.view(dtype=dtype)

    @parametrize("n,k", [(4096, 4096), (4096, 11008), (11008, 4096)])
    @pytest.mark.skipif(
        not torch.xpu.has_xetla(),
        reason="int4-dequantize kernel should compile with XeTLA",
    )
    def test_int4x8_dequant(self, n, k):
        qweight = self.rand_int4(k * n, torch.int32, "xpu").reshape(k // 8, n)

        group_size = min(128, k)
        group_num = int(k / group_size)

        scales = -torch.rand([group_num, n], device="xpu", dtype=torch.float16)
        zero_points = self.rand_int4(group_num * n, torch.int32, "xpu").reshape(
            group_num, n // 8
        )

        ref_weight = self.dequantize(qweight, scales, zero_points, group_size).cpu()
        tar_weight = torch.ops.torch_ipex.int4x8_dequantize(
            qweight.t().contiguous(), scales.t().contiguous(), None, group_size
        )
        self.assertEqual(
            tar_weight.cpu().float(),
            ref_weight.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )


instantiate_parametrized_tests(TestInt4Dequantize)

if __name__ == "__main__":
    run_tests()
