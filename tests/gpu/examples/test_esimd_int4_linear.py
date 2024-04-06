import torch

torch.set_printoptions(profile="full")
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

        # print(weight_fp16)

        # print("weight_fp16: line 00:", weight_fp16[0][0][0])
        # print("weight_fp16: line 01:", weight_fp16[0][0][1])
        # print("weight_fp16: line 10:", weight_fp16[0][1][0])
        if gemm_num == 1:
            weight_fp16 = weight_fp16.reshape([k, n])
        return weight_fp16

    # def test_gemm_int4(self, per_channel=False, dtype=torch.float16):
    #     input = torch.rand([1, 4096], device="xpu", dtype=torch.float16) - 0.5
    #     # for ind in range(2048, 4096):
    #     #     input[0][ind] = 0.0

    #     group_size = 32  # 128 -> 32
    #     if per_channel:
    #         group_size = 4096
    #     group_num = int(4096 / group_size)

    #     scales = (
    #         torch.rand([group_num, 32], device="xpu", dtype=torch.float16) - 0.5
    #     )
    #     # zero_points = torch.rand([group_num, 8192], device="xpu").byte()
    #     zero_points = torch.zeros(group_num, 16, device="xpu").byte()
    #     weight = (torch.rand([4096, 16], device="xpu") * 10).byte()
    #     #print(weight)

    #     weight_fp16 = self.dequantize(weight, scales, zero_points, group_size)

    #     # check gemm
    #     out_int4 = torch.ops.torch_ipex.mm_esimd_int4(
    #         input, weight, scales, zero_points, group_size
    #     )
    #     print("out_int4: ", out_int4.to(torch.float).to("cpu"))

    #     out_fp16 = torch.matmul(input, weight_fp16)
    #     print("out_fp16: ", out_fp16.to(torch.float).to("cpu"))
    #     self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)

    # def test_gemm_int4_sp_data(self, per_channel=False, dtype=torch.float16):
    #     input = torch.rand([1, 4096], device="xpu", dtype=torch.float16) * 0
    #     # input = torch.zeros(1, 4096, device="xpu", dtype=torch.float16) + 1
    #     input[0][100] = 2.0

    #     group_size = 32  # 128 -> 32
    #     if per_channel:
    #         group_size = 4096
    #     group_num = int(4096 / group_size)

    #     scales = torch.rand([group_num, 32], device="xpu", dtype=torch.float16) * 0
    #     scales[3][0] = 2.0
    #     scales[3][1] = 1.7
    #     # print("scales: ", scales.to(torch.float).to("cpu"))
    #     print(scales.stride())

    #     scales_t = torch.transpose(scales, 0, 1)
    #     print(scales_t.stride())
    #     # print("scales_t: ", scales_t.to(torch.float).to("cpu"))

    #     zero_points = (torch.rand([group_num, 16], device="xpu") * 0).byte()
    #     # weight = (torch.rand([4096, 16], device="xpu") * 0 + 17).byte()
    #     weight = (torch.rand([4096, 16], device="xpu") * 0).byte()
    #     weight[100][0] = 0x32

    #     weight_fp16 = self.dequantize(weight, scales, zero_points, group_size)

    #     # check gemm
    #     out_int4 = torch.ops.torch_ipex.mm_esimd_int4(
    #         input, weight, scales_t, zero_points, group_size
    #     )
    #     print("out_int4: ", out_int4.to(torch.float).to("cpu"))

    #     out_fp16 = torch.matmul(input, weight_fp16)
    #     print("out_fp16: ", out_fp16.to(torch.float).to("cpu"))
    #     self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)

    # def test_gemm_int4_ref_issue(self, per_channel=False, dtype=torch.float16):
    #     input = torch.ones(1, 4096, device="xpu", dtype=torch.float16)

    #     group_size = 32  # 128 -> 32
    #     if per_channel:
    #         group_size = 4096
    #     group_num = int(4096 / group_size)

    #     scales = (
    #         torch.ones(group_num, 32, device="xpu", dtype=torch.float16)
    #     )
    #     # zero_points = torch.rand([group_num, 8192], device="xpu").byte()
    #     zero_points = torch.zeros(group_num, 16, device="xpu").byte()
    #     weight = (torch.ones(4096, 16, device="xpu")).byte()

    #     weight_fp16 = self.dequantize(weight, scales, zero_points, group_size)

    #     # check gemm
    #     out_int4 = torch.ops.torch_ipex.mm_esimd_int4(
    #         input, weight, scales, zero_points, group_size
    #     )
    #     print("out_int4: ", out_int4.to(torch.float).to("cpu"))

    #     out_fp16 = torch.matmul(input, weight_fp16)
    #     print("out_fp16: ", out_fp16.to(torch.float).to("cpu"))
    #     self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)

    def test_gemm_int4_refcpu(self, per_channel=False, dtype=torch.float16):
        input = torch.rand([1, 4096], device="xpu", dtype=torch.float16)
        # for ind in range(2048, 4096):
        #     input[0][ind] = 0.0

        group_size = 32  # 128 -> 32
        if per_channel:
            group_size = 4096
        group_num = int(4096 / group_size)

        scales = torch.rand([group_num, 16384], device="xpu", dtype=torch.float16)
        # zero_points = torch.rand([group_num, 8192], device="xpu").byte()
        zero_points = torch.zeros(group_num, 8192, device="xpu").byte()
        weight = (torch.rand([4096, 8192], device="xpu") * 10 * 16 ).byte()
        # print(weight)

        weight_fp16 = self.dequantize(weight, scales, zero_points, group_size)

        # check gemm
        out_int4 = torch.ops.torch_ipex.mm_esimd_int4(
            input, weight, scales, zero_points, group_size
        )
        print("out_int4: ", out_int4.to(torch.float).to("cpu"))

        weight_fp32 = weight_fp16.to(torch.float).to("cpu")
        input_fp32 = input.to(torch.float).to("cpu")
        out_fp16 = torch.matmul(input_fp32, weight_fp32)
        print("out_fp16: ", out_fp16.to(torch.float).to("cpu"))

        out_int4 = out_int4.to(torch.float).to("cpu")
        self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)


# res 26: 2 * 2 *2 + 1 * 1.2 * 8 = 17.6
# res 27: 2 * 1.7 * 3 + 1 * 2.5 * 1 = 12.7

# 8 10.2
