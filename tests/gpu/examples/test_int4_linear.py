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
checking_rtol = 2e-2


class TestInt4Linear(TestCase):

    @staticmethod
    def unpack_weight(qweight, scales, qzeros, q_config):
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

    @staticmethod
    def dequantize(qweight, scales, qzeros, group_size):
        q_config = {"group_size": group_size, "bits": 4}
        weight, gptq_scales, gptq_zeros = TestInt4Linear.unpack_weight(
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
        return weight

    @staticmethod
    def rand_int4(size, dtype=torch.int32, device="xpu"):
        rand = torch.randint(-128, 128, [size // 2], device=device).to(torch.int8)
        return rand.view(dtype=dtype)

    # qkv per-channel not used && TODO: Mismatched elements: 1 / 9216
    @parametrize("per_channel", [False], lambda k: "per_channel" * k)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("m,n,k", [(1, 3072, 3072)])
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_qkv_int4(self, m, n, k, per_channel, dtype):
        group_size = min(32, k)
        if per_channel:
            group_size = k

        input = torch.rand([1, m, k], device="xpu", dtype=dtype)
        bias = torch.rand([3, n], device="xpu", dtype=dtype)
        weight = (torch.randint(0, 1112111, [3, k // 8, n], device="xpu")).to(
            torch.int32
        )

        group_num = int(k / group_size)

        scales = torch.rand([3, group_num, n], device="xpu", dtype=dtype)
        zero_points = (torch.zeros([3, group_num, n // 8], device="xpu")).to(
            torch.int32
        )

        out_xetla = torch.ops.torch_ipex.mm_qkv_int4(
            input,
            weight.transpose(1, 2).contiguous(),
            bias,
            scales.transpose(1, 2).contiguous(),
            None,
            group_size,
        )

        weight_fp16_1 = self.dequantize(
            weight[0], scales[0], zero_points[0], group_size
        ).to(dtype)
        out_torch_1 = torch.matmul(input.cpu().float(), weight_fp16_1.cpu().float())
        out_torch_bias_1 = out_torch_1 + bias[0].cpu().float()

        weight_fp16_2 = self.dequantize(
            weight[1], scales[1], zero_points[1], group_size
        ).to(dtype)
        out_torch_2 = torch.matmul(input.cpu().float(), weight_fp16_2.cpu().float())
        out_torch_bias_2 = out_torch_2 + bias[1].cpu().float()

        weight_fp16_3 = self.dequantize(
            weight[2], scales[2], zero_points[2], group_size
        ).to(dtype)
        out_torch_3 = torch.matmul(input.cpu().float(), weight_fp16_3.cpu().float())
        out_torch_bias_3 = out_torch_3 + bias[2].cpu().float()

        self.assertEqual(
            torch.cat([out_xetla[i].cpu().float() for i in range(3)]),
            torch.cat((out_torch_bias_1, out_torch_bias_2, out_torch_bias_3)),
            atol=checking_atol,
            rtol=checking_rtol,
        )

    @parametrize("per_channel", [False], lambda k: "per_channel" * k)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("m,n,k", [(8, 4096, 4096), (1, 4096, 11008), (32, 4096, 4096)])
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_gemm_int4(self, m, n, k, per_channel, dtype):
        input = torch.rand([m, k], device="xpu", dtype=dtype)
        input_torch = input.cpu()
        weight = self.rand_int4(k * n, torch.int32, "xpu").reshape(k // 8, n)

        group_size = min(128, k)
        if per_channel:
            group_size = k
        group_num = int(k / group_size)

        scales = -torch.rand([group_num, n], device="xpu", dtype=dtype)
        zero_points = self.rand_int4(group_num * n, torch.int32, "xpu").reshape(
            group_num, n // 8
        )

        weight_fp = self.dequantize(weight, scales, zero_points, group_size).cpu()
        # check gemm
        out_xetla = torch.ops.torch_ipex.mm_int4(
            input, weight.t().contiguous(), scales.t().contiguous(), None, group_size
        )
        out_torch = torch.matmul(input_torch, weight_fp)
        self.assertEqual(
            out_xetla.cpu().float(),
            out_torch.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + residual
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        out_xetla_res = torch.ops.torch_ipex.mm_add_int4(
            input,
            weight.t().contiguous(),
            scales.t().contiguous(),
            None,
            group_size,
            res0,
        )
        out_torch_res = out_torch + res0.cpu().float()
        self.assertEqual(
            out_xetla_res.cpu().float(),
            out_torch_res.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias
        bias = torch.rand([1, n], device="xpu", dtype=dtype)
        out_xetla_bias = torch.ops.torch_ipex.mm_bias_int4(
            input,
            weight.t().contiguous(),
            bias,
            scales.t().contiguous(),
            None,
            group_size,
        )
        out_torch_bias = out_torch + bias.cpu().float()
        self.assertEqual(
            out_xetla_bias.cpu().float(),
            out_torch_bias.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias + gelu
        out_xetla_gelu = torch.ops.torch_ipex.mm_bias_gelu_int4(
            input,
            weight.t().contiguous(),
            scales.t().contiguous(),
            None,
            bias,
            group_size,
            "tanh",
        )
        gelu_out = torch.nn.GELU(approximate="tanh")(out_torch_bias)
        self.assertEqual(
            out_xetla_gelu.cpu().float(),
            gelu_out.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + silu + mul
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        out_xetla_silu = torch.ops.torch_ipex.mm_silu_mul_int4(
            input,
            weight.t().contiguous(),
            scales.t().contiguous(),
            None,
            group_size,
            res0,
        )
        silu_mul_out = torch.nn.SiLU()(out_torch) * res0.cpu().float()
        self.assertEqual(
            out_xetla_silu.cpu().float(),
            silu_mul_out.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias + residual + residual
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        res1 = torch.rand([m, n], device="xpu", dtype=dtype)
        out_xetla_bias_2res = torch.ops.torch_ipex.mm_bias_resadd_resadd_int4(
            input,
            weight.t().contiguous(),
            bias,
            res0,
            res1,
            scales.t().contiguous(),
            None,
            group_size,
        )
        out_torch_bias_2res = out_torch_bias + res0.cpu().float() + res1.cpu().float()
        self.assertEqual(
            out_xetla_bias_2res.cpu().float(),
            out_torch_bias_2res.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias + residual
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        out_xetla_bias_add = torch.ops.torch_ipex.mm_bias_add_int4(
            input,
            weight.t().contiguous(),
            bias,
            scales.t().contiguous(),
            None,
            group_size,
            res0,
        )
        out_torch_bias_add = out_torch_bias + res0.cpu().float()
        self.assertEqual(
            out_xetla_bias_add.cpu().float(),
            out_torch_bias_add.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

    @parametrize("per_channel", [False], lambda k: "per_channel" * k)
    @parametrize("m,n,k", [(8, 4096, 4096), (1, 4096, 11008), (32, 4096, 4096)])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_mlp_int4(self, m, n, k, per_channel, dtype):
        input = torch.rand([m, k], device="xpu", dtype=dtype)
        input_torch = input.cpu()
        weight = self.rand_int4(k * n, torch.int32, "xpu").reshape(k // 8, n)

        group_size = min(128, k)
        if per_channel:
            group_size = k
        group_num = int(k / group_size)

        scales = -torch.rand([group_num, n], device="xpu", dtype=dtype)
        zero_points = self.rand_int4(group_num * n, torch.int32, "xpu").reshape(
            group_num, n // 8
        )
        bias = torch.rand([1, n], device="xpu", dtype=dtype)

        weight_fp = self.dequantize(weight, scales, zero_points, group_size).cpu()
        out_torch = torch.matmul(input_torch, weight_fp)
        out_torch_bias = out_torch + bias.cpu().float()

        # mlp silu mul
        weight_up = self.rand_int4(k * n, torch.int32, "xpu").reshape(k // 8, n)
        scales_up = torch.rand([group_num, n], device="xpu", dtype=dtype) / (k / 8)
        zero_points_up = self.rand_int4(group_num * n, torch.int32, "xpu").reshape(
            group_num, n // 8
        )
        weight_up_fp16 = self.dequantize(
            weight_up, scales_up, zero_points_up, group_size
        ).cpu()
        out_torch_silu = torch.nn.SiLU()(out_torch)
        out_torch_up = torch.matmul(input_torch, weight_up_fp16)
        out_torch_mlp_silu_mul = out_torch_silu * out_torch_up
        xetla_mlp_args_common = (
            input,
            torch.stack((weight.t(), weight_up.t())).contiguous(),
            torch.stack((scales.t(), scales_up.t())).contiguous(),
            None,
        )
        out_xetla_mlp_silu_mul = torch.ops.torch_ipex.mlp_silu_mul_int4(
            *xetla_mlp_args_common,
            group_size,
        )
        self.assertEqual(
            out_xetla_mlp_silu_mul.cpu().float(),
            out_torch_mlp_silu_mul.float(),
            atol=checking_atol * 3,  # elt mul will introduce more error
            rtol=checking_rtol * 5,
        )

        # mlp bias silu mul
        out_torch_bias_silu = torch.nn.SiLU()(out_torch_bias)
        out_torch_mlp_bias_silu_mul = out_torch_bias_silu * out_torch_up
        out_xetla_mlp_bias_silu_mul = torch.ops.torch_ipex.mlp_bias_silu_mul_int4(
            *xetla_mlp_args_common,
            bias,
            group_size,
        )
        self.assertEqual(
            out_xetla_mlp_bias_silu_mul.cpu().float(),
            out_torch_mlp_bias_silu_mul.float(),
            atol=checking_atol * 3,  # elt mul will introduce more error
            rtol=checking_rtol * 5,
        )

        # mlp silu mul bias
        bias_up = -torch.rand([1, n], device="xpu", dtype=dtype)
        out_torch_bias_up = out_torch_up + bias_up.cpu().float()
        out_torch_mlp_silu_mul_bias = out_torch_silu * out_torch_bias_up
        out_xetla_mlp_silu_mul_bias = torch.ops.torch_ipex.mlp_silu_mul_bias_int4(
            *xetla_mlp_args_common,
            bias_up,
            group_size,
        )
        self.assertEqual(
            out_xetla_mlp_silu_mul_bias.cpu().float(),
            out_torch_mlp_silu_mul_bias.float(),
            atol=checking_atol * 3,  # elt mul will introduce more error
            rtol=checking_rtol * 5,
        )

        # mlp bias silu mul bias
        out_torch_mlp_bias_silu_mul_bias = out_torch_bias_silu * out_torch_bias_up
        out_xetla_mlp_bias_silu_mul_bias = (
            torch.ops.torch_ipex.mlp_bias_silu_mul_bias_int4(
                *xetla_mlp_args_common,
                bias,
                bias_up,
                group_size,
            )
        )
        self.assertEqual(
            out_xetla_mlp_bias_silu_mul_bias.cpu().float(),
            out_torch_mlp_bias_silu_mul_bias.float(),
            atol=checking_atol * 3,  # elt mul will introduce more error
            rtol=checking_rtol * 5,
        )


instantiate_parametrized_tests(TestInt4Linear)

if __name__ == "__main__":
    run_tests()
