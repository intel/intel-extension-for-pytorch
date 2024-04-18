import torch

# torch.set_printoptions(profile="full")
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
        weight = (torch.rand([4096, 8192], device="xpu") * 10 * 16).byte()
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

    def test_gemm_int4_long_hidden_dim_refcpu(
        self, per_channel=False, dtype=torch.float16
    ):
        input = torch.rand([1, 11008], device="xpu", dtype=torch.float16)
        # for ind in range(2048, 4096):
        #     input[0][ind] = 0.0

        group_size = 32  # 128 -> 32
        if per_channel:
            group_size = 11008
        group_num = int(11008 / group_size)

        scales = torch.rand([group_num, 4096], device="xpu", dtype=torch.float16)
        # zero_points = torch.rand([group_num, 8192], device="xpu").byte()
        zero_points = torch.zeros(group_num, 2048, device="xpu").byte()
        weight = (torch.rand([11008, 2048], device="xpu") * 10 * 16).byte()
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

    def test_gemm_int4_long_hidden_dim_2_refcpu(
        self, per_channel=False, dtype=torch.float16
    ):
        input = torch.rand([1, 14336], device="xpu", dtype=torch.float16)
        # for ind in range(2048, 4096):
        #     input[0][ind] = 0.0

        group_size = 32  # 128 -> 32
        if per_channel:
            group_size = 14336
        group_num = int(14336 / group_size)

        scales = torch.rand([group_num, 4096], device="xpu", dtype=torch.float16)
        # zero_points = torch.rand([group_num, 8192], device="xpu").byte()
        zero_points = torch.zeros(group_num, 2048, device="xpu").byte()
        weight = (torch.rand([14336, 2048], device="xpu") * 10 * 16).byte()
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

    # def test_gemm_int4_long_hidden_dim_2_perf(
    #     self, per_channel=False, dtype=torch.float16
    # ):
    #     for ind in range(0, 4):
    #         input = torch.rand([1, 14336], device="xpu", dtype=torch.float16)
    #         # for ind in range(2048, 4096):
    #         #     input[0][ind] = 0.0

    #         group_size = 32  # 128 -> 32
    #         if per_channel:
    #             group_size = 14336
    #         group_num = int(14336 / group_size)

    #         scales = torch.rand([group_num, 4096], device="xpu", dtype=torch.float16)
    #         # zero_points = torch.rand([group_num, 8192], device="xpu").byte()
    #         zero_points = torch.zeros(group_num, 2048, device="xpu").byte()
    #         weight = (torch.rand([14336, 2048], device="xpu") * 10 * 16).byte()
    #         # print(weight)

    #         # check gemm
    #         out_int4 = torch.ops.torch_ipex.mm_esimd_int4(
    #             input, weight, scales, zero_points, group_size
    #         )
    #         print(
    #             "###################################################################################"
    #         )

    # def test_gemm_int4_long_hidden_dim_2_refcpu_spdata(
    #     self, per_channel=False, dtype=torch.float16
    # ):
    #     input = torch.rand([1, 14336], device="xpu", dtype=torch.float16) * 0 + 1
    #     # for ind in range(2048, 4096):
    #     #     input[0][ind] = 0.0

    #     group_size = 32  # 128 -> 32
    #     if per_channel:
    #         group_size = 14336
    #     group_num = int(14336 / group_size)

    #     scales = (
    #         torch.rand([group_num, 2048], device="xpu", dtype=torch.float16) * 0 + 1
    #     )
    #     # zero_points = torch.rand([group_num, 8192], device="xpu").byte()
    #     zero_points = torch.zeros(group_num, 1024, device="xpu").byte()
    #     weight = (torch.rand([14336, 1024], device="xpu") * 0 + 17).byte()
    #     # print(weight)

    #     weight_fp16 = self.dequantize(weight, scales, zero_points, group_size)

    #     # check gemm
    #     out_int4 = torch.ops.torch_ipex.mm_esimd_int4(
    #         input, weight, scales, zero_points, group_size
    #     )
    #     print("out_int4: ", out_int4.to(torch.float).to("cpu"))

    #     weight_fp32 = weight_fp16.to(torch.float).to("cpu")
    #     input_fp32 = input.to(torch.float).to("cpu")
    #     out_fp16 = torch.matmul(input_fp32, weight_fp32)
    #     print("out_fp16: ", out_fp16.to(torch.float).to("cpu"))

    #     out_int4 = out_int4.to(torch.float).to("cpu")
    #     self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)

    # def test_gemm_int4_perf(self, per_channel=False, dtype=torch.float16):
    #     for ind in range(0, 4):
    #         input = torch.rand([1, 4096], device="xpu", dtype=torch.float16)
    #         # for ind in range(2048, 4096):
    #         #     input[0][ind] = 0.0

    #         group_size = 32  # 128 -> 32
    #         if per_channel:
    #             group_size = 4096
    #         group_num = int(4096 / group_size)

    #         scales = torch.rand([group_num, 4096], device="xpu", dtype=torch.float16)
    #         # zero_points = torch.rand([group_num, 8192], device="xpu").byte()
    #         zero_points = torch.zeros(group_num, 2048, device="xpu").byte()
    #         weight = (torch.rand([4096, 2048], device="xpu") * 10 * 16).byte()
    #         # print(weight)

    #         # check gemm
    #         with torch.autograd.profiler_legacy.profile(
    #             enabled=True, use_xpu=True, record_shapes=True
    #         ) as prof:
    #             out_int4 = torch.ops.torch_ipex.mm_esimd_int4(
    #                 input, weight, scales, zero_points, group_size
    #             )
    #         # print("out_int4: ", out_int4.to(torch.float).to("cpu"))
    #         print(prof.key_averages().table(sort_by="self_xpu_time_total"))

    #         print(
    #             "###################################################################################"
    #         )

    # def test_gemm_int4_long_hidden_dim_perf(self, per_channel=False, dtype=torch.float16):
    #     for ind in range(0, 4):
    #         input = torch.rand([1, 11008], device="xpu", dtype=torch.float16)
    #         # for ind in range(2048, 4096):
    #         #     input[0][ind] = 0.0

    #         group_size = 32  # 128 -> 32
    #         if per_channel:
    #             group_size = 11008
    #         group_num = int(11008 / group_size)

    #         scales = torch.rand([group_num, 4096], device="xpu", dtype=torch.float16)
    #         # zero_points = torch.rand([group_num, 8192], device="xpu").byte()
    #         zero_points = torch.zeros(group_num, 2048, device="xpu").byte()
    #         weight = (torch.rand([4096, 2048], device="xpu") * 10 * 16).byte()
    #         # print(weight)

    #         # check gemm
    #         with torch.autograd.profiler_legacy.profile(
    #             enabled=True, use_xpu=True, record_shapes=True
    #         ) as prof:
    #             out_int4 = torch.ops.torch_ipex.mm_esimd_int4(
    #                 input, weight, scales, zero_points, group_size
    #             )
    #         # print("out_int4: ", out_int4.to(torch.float).to("cpu"))
    #         print(prof.key_averages().table(sort_by="self_xpu_time_total"))

    #         print(
    #             "###################################################################################"
    #         )

    # def test_gemm_int4_large_non_hidden_dim_perf(self, per_channel=False, dtype=torch.float16):
    #     for ind in range(0, 4):
    #         input = torch.rand([1, 4096], device="xpu", dtype=torch.float16)
    #         # for ind in range(2048, 4096):
    #         #     input[0][ind] = 0.0

    #         group_size = 32  # 128 -> 32
    #         if per_channel:
    #             group_size = 4096
    #         group_num = int(4096 / group_size)

    #         scales = torch.rand([group_num, 12288], device="xpu", dtype=torch.float16)
    #         # zero_points = torch.rand([group_num, 8192], device="xpu").byte()
    #         zero_points = torch.zeros(group_num, 6144, device="xpu").byte()
    #         weight = (torch.rand([4096, 6144], device="xpu") * 10 * 16).byte()
    #         # print(weight)

    #         # check gemm
    #         with torch.autograd.profiler_legacy.profile(
    #             enabled=True, use_xpu=True, record_shapes=True
    #         ) as prof:
    #             out_int4 = torch.ops.torch_ipex.mm_esimd_int4(
    #                 input, weight, scales, zero_points, group_size
    #             )
    #         # print("out_int4: ", out_int4.to(torch.float).to("cpu"))
    #         print(prof.key_averages().table(sort_by="self_xpu_time_total"))

    #         print(
    #             "###################################################################################"
    #         )

    def test_qkv_gemm_int4_using_normal_gemm_as_ref(self, dtype=torch.float16):
        input = torch.rand([1, 1, 4096], device="xpu", dtype=torch.float16)
        q = torch.zeros([1, 4096], device="xpu", dtype=torch.float16)
        k = torch.zeros([1, 4096], device="xpu", dtype=torch.float16)
        v = torch.zeros([1, 4096], device="xpu", dtype=torch.float16)

        bias = None
        weight = (torch.rand([3, 4096, 2048], device="xpu") * 10 * 16).byte()
        print("weight.size()", weight.size())
        print("weight.stride()", weight.stride())
        weight_ref = torch.clone(weight)

        group_size = 32
        group_num = int(4096 / group_size)

        scales = torch.rand([3, group_num, 4096], device="xpu", dtype=torch.float16)
        zero_points = (torch.zeros([3, group_num, 2048], device="xpu")).byte()

        print("scales.size()", scales.size())
        print("scales.stride()", scales.stride())
        scales_ref = torch.clone(scales)

        torch.ops.torch_ipex.qkv_mm_esimd_int4(
            input, weight, scales, zero_points, bias, q, k, v, group_size
        )

        print("q ", q.to(torch.float).to("cpu"))
        print("k ", k.to(torch.float).to("cpu"))
        print("v ", v.to(torch.float).to("cpu"))

        q_ref = torch.ops.torch_ipex.mm_esimd_int4(
            input[0], weight_ref[0], scales_ref[0], zero_points[0], group_size
        )
        k_ref = torch.ops.torch_ipex.mm_esimd_int4(
            input[0], weight_ref[1], scales_ref[1], zero_points[1], group_size
        )
        v_ref = torch.ops.torch_ipex.mm_esimd_int4(
            input[0], weight_ref[2], scales_ref[2], zero_points[2], group_size
        )

        print("q ref", q_ref.to(torch.float).to("cpu"))
        print("k ref", k_ref.to(torch.float).to("cpu"))
        print("v ref", v_ref.to(torch.float).to("cpu"))

        self.assertEqual(q, q_ref, atol=1e-3, rtol=1e-3)
        self.assertEqual(k, k_ref, atol=1e-3, rtol=1e-3)
        self.assertEqual(v, v_ref, atol=1e-3, rtol=1e-3)

    def test_qkv_gemm_int4_fused_using_normal_gemm_as_ref(self, dtype=torch.float16):
        input = torch.rand([1, 1, 4096], device="xpu", dtype=torch.float16)
        q = torch.zeros([1, 4096], device="xpu", dtype=torch.float16)
        k = torch.zeros([1, 1024], device="xpu", dtype=torch.float16)
        v = torch.zeros([1, 1024], device="xpu", dtype=torch.float16)

        bias = None
        weight = (torch.rand([1, 4096, 3072], device="xpu") * 10 * 16).byte()

        print("weight.size()", weight.size())
        print("weight.stride()", weight.stride())
        weight_0_ref = torch.clone(weight[0][:, :2048])
        weight_1_ref = torch.clone(weight[0][:, 2048 : 2048 + 512])
        weight_2_ref = torch.clone(weight[0][:, 2048 + 512 :])
        # print("weight_0_ref.size()", weight_0_ref.size())
        # print("weight_0_ref.stride()", weight_0_ref.stride())
        # print("weight_1_ref.size()", weight_1_ref.size())
        # print("weight_1_ref.stride()", weight_1_ref.stride())
        # print("weight_0_ref:\n", weight_0_ref)
        # print("weight_1_ref:\n", weight_1_ref)
        group_size = 32
        group_num = int(4096 / group_size)

        scales = torch.rand([1, group_num, 6144], device="xpu", dtype=torch.float16)
        zero_points = (torch.zeros([1, group_num, 3072], device="xpu")).byte()

        # print("scales.size()", scales.size())
        # print("scales.stride()", scales.stride())
        scales_0_ref = torch.clone(scales[0][:, :4096])
        scales_1_ref = torch.clone(scales[0][:, 4096 : (4096 + 1024)])
        scales_2_ref = torch.clone(scales[0][:, (4096 + 1024) :])
        # print("scales_0_ref.size()", scales_0_ref.size())
        # print("scales_0_ref.stride()", scales_0_ref.stride())
        # print("scales_0_ref:\n", scales_0_ref.to(torch.float).to("cpu"))
        # print("scales_1_ref.size()", scales_1_ref.size())
        # print("scales_1_ref.stride()", scales_1_ref.stride())
        # print("scales_1_ref:\n", scales_1_ref.to(torch.float).to("cpu"))

        torch.ops.torch_ipex.qkv_mm_esimd_int4(
            input, weight, scales, zero_points, bias, q, k, v, group_size, True
        )

        print("q ", q.to(torch.float).to("cpu"))
        print("k ", k.to(torch.float).to("cpu"))
        print("v ", v.to(torch.float).to("cpu"))

        q_ref = torch.ops.torch_ipex.mm_esimd_int4(
            input[0],
            weight_0_ref,
            scales_0_ref,
            zero_points[0][:, :2048],
            group_size,
            True,
        )
        k_ref = torch.ops.torch_ipex.mm_esimd_int4(
            input[0],
            weight_1_ref,
            scales_1_ref,
            zero_points[0][:, 2048 : (2048 + 512)],
            group_size,
            True,
        )
        v_ref = torch.ops.torch_ipex.mm_esimd_int4(
            input[0],
            weight_2_ref,
            scales_2_ref,
            zero_points[0][:, (2048 + 512) :],
            group_size,
            True,
        )

        print("q ref", q_ref.to(torch.float).to("cpu"))
        print("k ref", k_ref.to(torch.float).to("cpu"))
        print("v ref", v_ref.to(torch.float).to("cpu"))

        self.assertEqual(q, q_ref, atol=1e-3, rtol=1e-3)
        self.assertEqual(k, k_ref, atol=1e-3, rtol=1e-3)
        self.assertEqual(v, v_ref, atol=1e-3, rtol=1e-3)

    # res 26: 2 * 2 *2 + 1 * 1.2 * 8 = 17.6
    # res 27: 2 * 1.7 * 3 + 1 * 2.5 * 1 = 12.7

    # 8 10.2

    # 1st token GEMMs ===========================================================================================

    # def test_gemm_1stt_int4_refcpu_debug0(self, per_channel=False, dtype=torch.float16):
    #     print("\n!!!!!test_gemm_1stt_int4_refcpu\n")
    #     input = torch.ones([2, 4096], device="xpu", dtype=torch.float16)

    #     group_size = 32  # 128 -> 32
    #     if per_channel:
    #         group_size = 4096
    #     group_num = int(4096 / group_size)

    #     scales = torch.ones([group_num, 4096], device="xpu", dtype=torch.float16)

    #     zero_points = torch.zeros(group_num, 2048, device="xpu").byte()
    #     weight = (torch.full([4096, 2048], 17, device="xpu")).byte()
    #     # print(weight)

    #     weight_fp16 = self.dequantize(weight, scales, zero_points, group_size)
    #     print("\n!!!!!dequant done\n")
    #     # check gemm
    #     out_int4 = torch.ops.torch_ipex.mm_esimd_int4(
    #         input, weight, scales, zero_points, group_size
    #     )
    #     print("out_int4: ", out_int4.to(torch.float).to("cpu"))

    #     weight_fp32 = weight_fp16.to(torch.float).to("cpu")
    #     input_fp32 = input.to(torch.float).to("cpu")
    #     out_fp16 = torch.matmul(input_fp32, weight_fp32)
    #     print("out_fp16: ", out_fp16.to(torch.float).to("cpu"))

    #     out_int4 = out_int4.to(torch.float).to("cpu")
    #     self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)

    # out_fp16 = torch.matmul(input, weight_fp16)

    #     print("out_fp16: ", out_fp16.to(torch.float).to("cpu"))

    # def test_gemm_1stt_int4_refcpu_debug1(self, per_channel=False, dtype=torch.float16):
    #     print("\n!!!!!test_gemm_1stt_int4_refcpu\n")
    #     input = torch.rand([64, 4096], device="xpu", dtype=torch.float16)

    #     group_size = 32  # 128 -> 32
    #     if per_channel:
    #         group_size = 4096
    #     group_num = int(4096 / group_size)

    #     scales = torch.rand([group_num, 4096], device="xpu", dtype=torch.float16)

    #     zero_points = torch.zeros(group_num, 2048, device="xpu").byte()
    #     weight = (torch.full([4096, 2048], 17, device="xpu")).byte()
    #     # print(weight)

    #     weight_fp16 = self.dequantize(weight, scales, zero_points, group_size)
    #     print("\n!!!!!dequant done\n")
    #     # check gemm
    #     out_int4 = torch.ops.torch_ipex.mm_esimd_int4(
    #         input, weight, scales, zero_points, group_size
    #     )
    #     print("out_int4: ", out_int4.to(torch.float).to("cpu"))

    #     weight_fp32 = weight_fp16.to(torch.float).to("cpu")
    #     input_fp32 = input.to(torch.float).to("cpu")
    #     out_fp16 = torch.matmul(input_fp32, weight_fp32)
    #     print("out_fp16: ", out_fp16.to(torch.float).to("cpu"))

    #     out_int4 = out_int4.to(torch.float).to("cpu")
    #     self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)

    def test_gemm_1stt_int4_refcpu(self, per_channel=False, dtype=torch.float16):
        print("\n!!!!!test_gemm_1stt_int4_refcpu\n")
        input = torch.rand([1024, 4096], device="xpu", dtype=torch.float16)

        group_size = 32  # 128 -> 32
        if per_channel:
            group_size = 4096
        group_num = int(4096 / group_size)

        scales = torch.rand([group_num, 4096], device="xpu", dtype=torch.float16)

        zero_points = torch.zeros(group_num, 2048, device="xpu").byte()
        weight = (torch.rand([4096, 2048], device="xpu") * 10 * 16).byte()
        # print(weight)

        weight_fp16 = self.dequantize(weight, scales, zero_points, group_size)
        print("\n!!!!!dequant done\n")
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

    def test_gemm_1stt_int4_long_hidden2_refcpu(
        self, per_channel=False, dtype=torch.float16
    ):
        print("\n!!!!!test_gemm_1stt_int4_refcpu\n")
        input = torch.rand([1024, 14336], device="xpu", dtype=torch.float16)

        group_size = 32  # 128 -> 32
        if per_channel:
            group_size = 14336
        group_num = int(14336 / group_size)

        scales = torch.rand([group_num, 4096], device="xpu", dtype=torch.float16)

        zero_points = torch.zeros(group_num, 2048, device="xpu").byte()
        weight = (torch.rand([14336, 2048], device="xpu") * 10 * 16).byte()
        # print(weight)

        weight_fp16 = self.dequantize(weight, scales, zero_points, group_size)
        print("\n!!!!!dequant done\n")
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

    def test_gemm_1stt_int4_long_hidden1_refcpu(
        self, per_channel=False, dtype=torch.float16
    ):
        print("\n!!!!!test_gemm_1stt_int4_refcpu\n")
        input = torch.rand([1024, 11008], device="xpu", dtype=torch.float16)

        group_size = 32  # 128 -> 32
        if per_channel:
            group_size = 11008
        group_num = int(11008 / group_size)

        scales = torch.rand([group_num, 4096], device="xpu", dtype=torch.float16)

        zero_points = torch.zeros(group_num, 2048, device="xpu").byte()
        weight = (torch.rand([11008, 2048], device="xpu") * 10 * 16).byte()
        # print(weight)

        weight_fp16 = self.dequantize(weight, scales, zero_points, group_size)
        print("\n!!!!!dequant done\n")
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

    def test_qkv_gemm_1stt_int4_using_normal_gemm_as_ref(self, dtype=torch.float16):
        input = torch.rand([1, 1024, 4096], device="xpu", dtype=torch.float16)
        q = torch.zeros([1024, 4096], device="xpu", dtype=torch.float16)
        k = torch.zeros([1024, 4096], device="xpu", dtype=torch.float16)
        v = torch.zeros([1024, 4096], device="xpu", dtype=torch.float16)

        bias = None
        weight = (torch.rand([3, 4096, 2048], device="xpu") * 10 * 16).byte()
        print("weight.size()", weight.size())
        print("weight.stride()", weight.stride())
        weight_ref = torch.clone(weight)

        group_size = 32
        group_num = int(4096 / group_size)

        scales = torch.rand([3, group_num, 4096], device="xpu", dtype=torch.float16)
        zero_points = (torch.zeros([3, group_num, 2048], device="xpu")).byte()

        print("scales.size()", scales.size())
        print("scales.stride()", scales.stride())
        scales_ref = torch.clone(scales)

        torch.ops.torch_ipex.qkv_mm_esimd_int4(
            input, weight, scales, zero_points, bias, q, k, v, group_size
        )

        print("q ", q.to(torch.float).to("cpu"))
        print("k ", k.to(torch.float).to("cpu"))
        print("v ", v.to(torch.float).to("cpu"))

        q_ref = torch.ops.torch_ipex.mm_esimd_int4(
            input[0], weight_ref[0], scales_ref[0], zero_points[0], group_size
        )
        k_ref = torch.ops.torch_ipex.mm_esimd_int4(
            input[0], weight_ref[1], scales_ref[1], zero_points[1], group_size
        )
        v_ref = torch.ops.torch_ipex.mm_esimd_int4(
            input[0], weight_ref[2], scales_ref[2], zero_points[2], group_size
        )

        print("q ref", q_ref.to(torch.float).to("cpu"))
        print("k ref", k_ref.to(torch.float).to("cpu"))
        print("v ref", v_ref.to(torch.float).to("cpu"))

        self.assertEqual(q, q_ref, atol=1e-3, rtol=1e-3)
        self.assertEqual(k, k_ref, atol=1e-3, rtol=1e-3)
        self.assertEqual(v, v_ref, atol=1e-3, rtol=1e-3)