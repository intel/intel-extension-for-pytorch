import torch
import intel_extension_for_pytorch  # noqa
import pytest
from intel_extension_for_pytorch.nn.utils._quantize_convert import GPTQShuffle

from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


checking_atol = 1e-2
checking_rtol = 1e-2
skip_bf16_input = not torch.xpu.has_2d_block_array() and not torch.xpu.has_xmx()


class TestOneDNNInt4Linear(TestCase):

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
    def dequantize(qweight, scales, qzeros, group_size, g_idx=None):
        q_config = {"group_size": group_size, "bits": 4}
        weight, gptq_scales, gptq_zeros = TestOneDNNInt4Linear.unpack_weight(
            qweight, scales, qzeros, q_config
        )
        gptq_zeros = (torch.ones_like(gptq_zeros) * 8).to("xpu")  # TODO: hard code zp
        if len(weight.shape) > 2:
            weight = weight.reshape(-1, weight.shape[-1])
        infeatures = weight.shape[0]
        if g_idx is None:
            g_idx = torch.tensor(
                [i // q_config["group_size"] for i in range(infeatures)],
                dtype=torch.int32,
            )
        scale_zeros = gptq_zeros * gptq_scales
        weight = gptq_scales[g_idx.long()] * weight - scale_zeros[g_idx.long()]
        return weight

    @staticmethod
    def rand_int4(size, dtype=torch.int32, device="xpu"):
        rand = torch.randint(-128, 128, [size // 2], device=device).to(torch.int8)
        return rand.view(dtype=dtype)

    @parametrize("per_channel", [False], lambda k: "per_channel" * k)
    @parametrize("dtype", [torch.float16])
    @parametrize("act_order", [False, True])
    @parametrize("m,n,k", [(8, 4096, 4096), (1, 4096, 11008), (32, 4096, 4096)])
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_gemm_int4(self, m, n, k, per_channel, act_order, dtype):
        input = torch.rand([m, k], device="xpu", dtype=dtype)
        input_torch = input.cpu()
        weight = self.rand_int4(k * n, torch.int32, "xpu").reshape(k // 8, n)

        group_size = min(128, k)
        if per_channel:
            group_size = k
        group_num = int(k / group_size)

        scales = torch.rand([group_num, n], device="xpu", dtype=dtype)
        zero_points = self.rand_int4(group_num * n, torch.int32, "xpu").reshape(
            group_num, n // 8
        )

        if act_order:
            g_idx = torch.randperm(k, dtype=torch.int32) // group_size
            shuf_weight = GPTQShuffle(bits=4, blocksize=group_size)
            shuffled_weight, g_idx4kernel = shuf_weight(weight, g_idx)
        else:
            g_idx = None
            g_idx4kernel = None
            shuffled_weight = weight

        weight_fp = self.dequantize(
            weight, scales, zero_points, group_size, g_idx
        ).cpu()
        # check gemm
        zero_points = torch.Tensor([8]).to(torch.int8).to("xpu")
        weight_ba = shuffled_weight.transpose(0, 1).contiguous().transpose(0, 1)

        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.ONEDNN):
            out_onednn = torch.ops.torch_ipex.mm_int4(
                input, weight_ba, scales, zero_points, group_size, g_idx4kernel
            )
        out_torch = torch.matmul(input_torch, weight_fp)
        self.assertEqual(
            out_onednn.cpu().float(),
            out_torch.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )
        # check gemm + residual
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.ONEDNN):
            out_onednn_res = torch.ops.torch_ipex.mm_add_int4(
                input, weight_ba, scales, zero_points, group_size, res0, g_idx4kernel
            )
        out_torch_res = out_torch + res0.cpu().float()
        self.assertEqual(
            out_onednn_res.cpu().float(),
            out_torch_res.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias
        bias = torch.rand([1, n], device="xpu", dtype=dtype)
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.ONEDNN):
            out_onednn_bias = torch.ops.torch_ipex.mm_bias_int4(
                input, weight_ba, bias, scales, zero_points, group_size, g_idx4kernel
            )
        out_torch_bias = out_torch + bias.cpu().float()
        self.assertEqual(
            out_onednn_bias.cpu().float(),
            out_torch_bias.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias + gelu
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.ONEDNN):
            out_onednn_gelu = torch.ops.torch_ipex.mm_bias_gelu_int4(
                input,
                weight_ba,
                scales,
                zero_points,
                bias,
                group_size,
                "tanh",
                g_idx4kernel,
            )
        gelu_out = torch.nn.GELU(approximate="tanh")(out_torch_bias)
        self.assertEqual(
            out_onednn_gelu.cpu().float(),
            gelu_out.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + silu + mul
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.ONEDNN):
            out_onednn_silu = torch.ops.torch_ipex.mm_silu_mul_int4(
                input, weight_ba, scales, zero_points, group_size, res0, g_idx4kernel
            )
        silu_mul_out = torch.nn.SiLU()(out_torch) * res0.cpu().float()
        self.assertEqual(
            out_onednn_silu.cpu().float(),
            silu_mul_out.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias + residual + residual
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        res1 = torch.rand([m, n], device="xpu", dtype=dtype)
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.ONEDNN):
            out_onednn_bias_2res = torch.ops.torch_ipex.mm_bias_resadd_resadd_int4(
                input,
                weight_ba,
                bias,
                res0,
                res1,
                scales,
                zero_points,
                group_size,
                g_idx4kernel,
            )
        out_torch_bias_2res = out_torch_bias + res0.cpu().float() + res1.cpu().float()
        self.assertEqual(
            out_onednn_bias_2res.cpu().float(),
            out_torch_bias_2res.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias + residual
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.ONEDNN):
            out_onednn_bias_add = torch.ops.torch_ipex.mm_bias_add_int4(
                input,
                weight_ba,
                bias,
                scales,
                zero_points,
                group_size,
                res0,
                g_idx4kernel,
            )
        out_torch_bias_add = out_torch_bias + res0.cpu().float()
        self.assertEqual(
            out_onednn_bias_add.cpu().float(),
            out_torch_bias_add.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )


instantiate_parametrized_tests(TestOneDNNInt4Linear)

if __name__ == "__main__":
    run_tests()
