import torch
import intel_extension_for_pytorch  # noqa

from torch.testing._internal.common_utils import TestCase

checking_atol = 1e-2
checking_rtol = 1e-2


class TestTorchMethod(TestCase):
    def dequantize(self, weight, scales, zero_points, group_size, gemm_num=1):
        k = weight.size()[-2]
        n = weight.size()[-1]
        weight = weight.reshape([gemm_num, k, n])
        n = n * 2

        group_num = int(k / group_size)

        scales = scales.reshape([gemm_num, group_num, n])
        zero_points = zero_points.reshape([gemm_num, group_num, int(n / 2)])

        weight_even = (((weight & 0x0F).to(torch.int8) << 4) >> 4).to(torch.int8)
        weight_odd = (weight.to(torch.int8) >> 4).to(torch.int8)

        zp_even = (zero_points & 0x0F).to(torch.int8)
        zp_even += 1
        zp_odd = (zero_points >> 4).to(torch.int8)
        zp_odd += 1

        weight_fp16 = []
        zp_fp16 = []
        for ind in range(0, n):
            if ind % 2 == 0:
                weight_fp16.append(
                    weight_even[:, :, int(ind / 2)].reshape([gemm_num, k, 1])
                )
                zp_fp16.append(
                    zp_even[:, :, int(ind / 2)].reshape([gemm_num, group_num, 1])
                )
            else:
                weight_fp16.append(
                    weight_odd[:, :, int(ind / 2)].reshape([gemm_num, k, 1])
                )
                zp_fp16.append(
                    zp_odd[:, :, int(ind / 2)].reshape([gemm_num, group_num, 1])
                )

        weight_fp16 = torch.concat(weight_fp16, dim=2)
        zp_fp16 = torch.concat(zp_fp16, dim=2)

        scales = torch.reshape(scales, [gemm_num, group_num, 1, n])
        zp_fp16 = torch.reshape(zp_fp16, [gemm_num, group_num, 1, n])
        scales = scales.repeat([1, 1, group_size, 1])
        zp_fp16 = zp_fp16.repeat([1, 1, group_size, 1])
        scales = torch.reshape(scales, [gemm_num, k, n])
        zp_fp16 = torch.reshape(zp_fp16, [gemm_num, k, n])

        weight_fp16 = (weight_fp16.to(torch.float16)) * scales
        if gemm_num == 1:
            weight_fp16 = weight_fp16.reshape([k, n])
        return weight_fp16

    def test_gemm_int4(self, per_channel=False, dtype=torch.float16):
        input = torch.rand([1, 4096], device="xpu", dtype=torch.float16)

        group_size = 128
        if per_channel:
            group_size = 4096
        group_num = int(4096 / group_size)

        scales = torch.rand([group_num, 16384], device="xpu", dtype=torch.float16)
        zero_points = torch.rand([group_num, 8192], device="xpu").byte()
        weight = (torch.rand([4096, 8192], device="xpu") * 10).byte()

        weight_fp16 = self.dequantize(weight, scales, zero_points, group_size)

        # check gemm
        out_int4 = torch.ops.torch_ipex.mm_esimd_int4(
            input, weight, scales, zero_points, group_size
        )
        out_fp16 = torch.matmul(input, weight_fp16)
        self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)
