import torch
import intel_extension_for_pytorch  # noqa
import pytest

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

        weight_even = (weight & 0x0f).to(torch.int8)
        weight_odd = (weight >> 4).to(torch.int8)

        zp_even = (zero_points & 0x0f).to(torch.int8)
        zp_even += 1
        zp_odd = (zero_points >> 4).to(torch.int8)
        zp_odd += 1

        weight_fp16 = []
        zp_fp16 = []
        for ind in range(0, n):
            if ind % 2 == 0:
                weight_fp16.append(weight_even[:, :, int(ind / 2)].reshape([gemm_num, k, 1]))
                zp_fp16.append(zp_even[:, :, int(ind / 2)].reshape([gemm_num, group_num, 1]))
            else:
                weight_fp16.append(weight_odd[:, :, int(ind / 2)].reshape([gemm_num, k, 1]))
                zp_fp16.append(zp_odd[:, :, int(ind / 2)].reshape([gemm_num, group_num, 1]))

        weight_fp16 = torch.concat(weight_fp16, dim=2)
        zp_fp16 = torch.concat(zp_fp16, dim=2)

        scales = torch.reshape(scales, [gemm_num, group_num, 1, n])
        zp_fp16 = torch.reshape(zp_fp16, [gemm_num, group_num, 1, n])
        scales = scales.repeat([1, 1, group_size, 1])
        zp_fp16 = zp_fp16.repeat([1, 1, group_size, 1])
        scales = torch.reshape(scales, [gemm_num, k, n])
        zp_fp16 = torch.reshape(zp_fp16, [gemm_num, k, n])

        weight_fp16 = ((weight_fp16 - zp_fp16).to(torch.float16)) * scales
        if gemm_num == 1:
            weight_fp16 = weight_fp16.reshape([k, n])
        return weight_fp16

    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_qkv_gemm_int4(self, per_channel=False, dtype=torch.float16):
        input = torch.rand([3, 4096, 4096], device="xpu", dtype=torch.float16)
        bias = torch.rand([3, 16384], device="xpu", dtype=torch.float16)
        weight = (torch.rand([3, 4096, 8192], device="xpu") * 10).byte()
        weight = weight.fill_(1)

        group_size = 128
        if per_channel:
            group_size = 4096
        group_num = int(4096 / group_size)

        scales = torch.ones([3, group_num, 16384], device="xpu", dtype=torch.float16)
        zero_points = (torch.zeros([3, group_num, 8192], device="xpu")).byte()

        out_int4 = torch.ops.torch_ipex.mm_qkv_int4(input, weight, bias, scales, zero_points, group_size)

        weight_fp16 = self.dequantize(weight, scales, zero_points, group_size, gemm_num=3)
        out_fp16 = torch.ops.torch_ipex.mm_qkv(input, weight_fp16, bias)

        self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)

    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_gemm_int4(self, per_channel=False, dtype=torch.float16):
        input = torch.rand([1, 4096], device="xpu", dtype=torch.float16)
        bias = torch.rand([16384], device="xpu", dtype=torch.float16)

        group_size = 128
        if per_channel:
            group_size = 4096
        group_num = int(4096 / group_size)

        scales = torch.rand([group_num, 16384], device="xpu", dtype=torch.float16)
        zero_points = torch.rand([group_num, 8192], device="xpu").byte()
        weight = torch.rand([4096, 8192], device="xpu").byte()

        weight_fp16 = self.dequantize(weight, scales, zero_points, group_size)

        # check gemm
        out_int4 = torch.ops.torch_ipex.mm_int4(input, weight, scales, zero_points, group_size)
        out_fp16 = torch.matmul(input, weight_fp16)
        self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)

        # check gemm + bias
        out_int4 = torch.ops.torch_ipex.mm_bias_int4(input, weight, bias, scales, zero_points, group_size)
        out_fp16 = out_fp16 + bias
        self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)

        # check gemm + bias + gelu
        out_int4 = torch.ops.torch_ipex.mm_bias_gelu_int4(input, weight, scales, zero_points, bias, group_size, "tanh")
        gelu_op = torch.nn.GELU(approximate="tanh")
        gelu_out = gelu_op(out_fp16)
        self.assertEqual(out_int4, gelu_out, atol=checking_atol, rtol=checking_rtol)

        # check gemm + bias + residual + residual
        res0 = torch.rand([1, 16384], device="xpu", dtype=torch.float16)
        res1 = torch.rand([1, 16384], device="xpu", dtype=torch.float16)
        out_int4 = torch.ops.torch_ipex.mm_bias_resadd_resadd_int4(input, weight, bias, res0, res1, scales, zero_points, group_size)
        out_fp16 = out_fp16 + res0 + res1
        self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)

    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_qkv_gemm_int4_per_channel(self, per_channel=True, dtype=torch.float16):
        input = torch.rand([3, 4096, 4096], device="xpu", dtype=torch.float16)
        bias = torch.rand([3, 16384], device="xpu", dtype=torch.float16)
        weight = (torch.rand([3, 4096, 8192], device="xpu") * 10).byte()
        weight = weight.fill_(1)

        group_size = 128
        if per_channel:
            group_size = 4096
        group_num = int(4096 / group_size)

        scales = torch.ones([3, group_num, 16384], device="xpu", dtype=torch.float16)
        zero_points = (torch.zeros([3, group_num, 8192], device="xpu")).byte()

        out_int4 = torch.ops.torch_ipex.mm_qkv_int4(input, weight, bias, scales, zero_points, group_size)

        weight_fp16 = self.dequantize(weight, scales, zero_points, group_size, gemm_num=3)
        out_fp16 = torch.ops.torch_ipex.mm_qkv(input, weight_fp16, bias)

        self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)

    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_gemm_int4_per_channel(self, per_channel=True, dtype=torch.float16):
        input = torch.rand([1, 4096], device="xpu", dtype=torch.float16)
        bias = torch.rand([16384], device="xpu", dtype=torch.float16)

        group_size = 128
        if per_channel:
            group_size = 4096
        group_num = int(4096 / group_size)

        scales = torch.rand([group_num, 16384], device="xpu", dtype=torch.float16)
        zero_points = torch.rand([group_num, 8192], device="xpu").byte()
        weight = torch.rand([4096, 8192], device="xpu").byte()

        weight_fp16 = self.dequantize(weight, scales, zero_points, group_size)

        # check gemm
        out_int4 = torch.ops.torch_ipex.mm_int4(input, weight, scales, zero_points, group_size)
        out_fp16 = torch.matmul(input, weight_fp16)
        self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)

        # check gemm + bias
        out_int4 = torch.ops.torch_ipex.mm_bias_int4(input, weight, bias, scales, zero_points, group_size)
        out_fp16 = out_fp16 + bias
        self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)

        # check gemm + bias + gelu
        out_int4 = torch.ops.torch_ipex.mm_bias_gelu_int4(input, weight, scales, zero_points, bias, group_size, "tanh")
        gelu_op = torch.nn.GELU(approximate="tanh")
        gelu_out = gelu_op(out_fp16)
        self.assertEqual(out_int4, gelu_out, atol=checking_atol, rtol=checking_rtol)

        # check gemm + bias + residual + residual
        res0 = torch.rand([1, 16384], device="xpu", dtype=torch.float16)
        res1 = torch.rand([1, 16384], device="xpu", dtype=torch.float16)
        out_int4 = torch.ops.torch_ipex.mm_bias_resadd_resadd_int4(input, weight, bias, res0, res1, scales, zero_points, group_size)
        out_fp16 = out_fp16 + res0 + res1
        self.assertEqual(out_int4, out_fp16, atol=checking_atol, rtol=checking_rtol)
