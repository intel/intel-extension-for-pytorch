import torch
import intel_extension_for_pytorch as ipex  # noqa
import pytest
from intel_extension_for_pytorch.nn.utils._quantize_convert import GPTQShuffle

from enum import Enum
from typing import List

from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


class QuantMode(Enum):
    SYM = 1
    ASYM = 2
    ASYM_FP_ZP = 3


base_atol = 1e-2
base_rtol = 2e-2
skip_bf16_input = not torch.xpu.has_2d_block_array() and not torch.xpu.has_xmx()
COMPILER_VERSION = torch.xpu.get_compiler_version()

AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def pack(imatrix: torch.Tensor, direction: str = "column"):
    """
    Packs a 4-bit integer matrix into a packed 32-bit integer matrix.
    Args:
        imatrix (torch.Tensor): matrix of integers
        direction (str): direction of packing, either "column" or "row"
    Returns:
        qmatrix (torch.Tensor): packed matrix of integers
    """
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=imatrix.device)

    imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

    if direction == "column":
        imatrix = imatrix.view(-1, imatrix.shape[1] // (32 // 4), (32 // 4))
        qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, None, :]).sum(dim=-1)

    elif direction == "row":
        imatrix = imatrix.view(imatrix.shape[0] // (32 // 4), (32 // 4), -1)
        qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, :, None]).sum(dim=1)

    qmatrix = qmatrix.to(torch.int32)

    return qmatrix


def unpack(qmatrix: torch.Tensor, direction: str = "column"):
    """
    Unpacks a 32-bit packed integer matrix into a 4-bit integer matrix.
    Args:
        qmatrix (torch.Tensor): matrix of packed integers
        direction (str): direction of unpacking, either "column" or "row"
    Returns:
        imatrix (torch.Tensor): matrix of integers
    """
    shifts = torch.arange(0, 32, 4, device=qmatrix.device)

    if direction == "column":
        imatrix = torch.bitwise_right_shift(
            qmatrix[:, :, None], shifts[None, None, :]
        ).view(qmatrix.shape[0], -1)

    elif direction == "row":
        imatrix = torch.bitwise_right_shift(
            qmatrix[:, None, :], shifts[None, :, None]
        ).view(-1, qmatrix.shape[-1])

    imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

    return imatrix


def apply_order(
    imatrix: torch.Tensor,
    direction: str = "column",
    order: List[int] = AWQ_PACK_ORDER,
):
    """
    Applies the order to a 4-bit integer matrix.
    Args:
        imatrix (torch.Tensor): matrix of integers
        direction (str): direction of applying order, either "column" or "row"
        order (List[int]): order to apply, default is AWQ_PACK_ORDER
    Returns:
        imatrix (torch.Tensor): matrix of integers
    """
    if direction == "column":
        imatrix = imatrix.view(-1, (32 // 4))[:, order].view(imatrix.shape)
    elif direction == "row":
        imatrix = imatrix.view((32 // 4), -1)[order, :].view(imatrix.shape)

    return imatrix


def fast_awq_to_gptq(qweight, qzeros):
    # awq uses column packing for both weights and zeros
    izeros = unpack(qzeros, direction="column")
    iweights = unpack(qweight, direction="column")

    # Reverse the order of the iweight and izeros tensors
    izeros = apply_order(izeros, direction="column", order=REVERSE_AWQ_PACK_ORDER)
    iweights = apply_order(iweights, direction="column", order=REVERSE_AWQ_PACK_ORDER)

    # exllama uses row packing for weights and column packing for zeros
    qzeros = pack(izeros, direction="column")
    qweight = pack(iweights, direction="row")

    return qweight, qzeros


def fast_gptq_to_awq(qweight, qzeros):
    # gptq uses row packing for both weights and zeros
    izeros = unpack(qzeros, direction="column")
    iweight = unpack(qweight, direction="row")

    izeros = apply_order(izeros, direction="column", order=AWQ_PACK_ORDER)
    iweight = apply_order(iweight, direction="row", order=AWQ_PACK_ORDER)

    izeros = izeros + 1

    qzeros = pack(izeros, direction="column")
    qweight = pack(iweight, direction="column")

    return qweight, qzeros


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
        zeros = qzeros
        if qzeros is not None and not qzeros.dtype.is_floating_point:
            zeros = torch.bitwise_right_shift(
                torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
            ).to(torch.int16 if bits == 8 else torch.int8)
            torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)

            # zeros = zeros + 1  # TODO(Yi): confirm dequant logic
            zeros = zeros.reshape(scales.shape)

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)
        ).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(weight, (2**bits) - 1, out=weight)

        return weight, scales, zeros

    @staticmethod
    def dequantize(qweight, scales, qzeros, group_size, g_idx=None):
        q_config = {"group_size": group_size, "bits": 4}
        weight, gptq_scales, gptq_zeros = TestInt4Linear.unpack_weight(
            qweight, scales, qzeros, q_config
        )
        if len(weight.shape) > 2:
            weight = weight.reshape(-1, weight.shape[-1])
        infeatures = weight.shape[0]
        if g_idx is None:
            g_idx = g_idx = (
                torch.arange(infeatures, dtype=torch.int32) // q_config["group_size"]
            )
        if gptq_zeros is None:
            return (weight - 8) * gptq_scales[g_idx]
        elif gptq_zeros.dtype.is_floating_point:
            return (weight - 8) * gptq_scales[g_idx] + gptq_zeros[g_idx]
        else:
            return (weight - gptq_zeros[g_idx]) * gptq_scales[g_idx]

    @staticmethod
    def rand_int4(size, dtype=torch.int32, device="xpu"):
        rand = torch.randint(-128, 128, [size // 2], device=device).to(torch.int8)
        return rand.view(dtype=dtype)

    # qkv per-channel not used && TODO: Mismatched elements: 1 / 9216
    @parametrize("per_channel", [False], lambda k: "per_channel" * k)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("qmode", list(QuantMode), lambda k: k.name)
    @parametrize(
        "m,n_q, n_kv ,k",
        [(1, 3072, 3072, 3072), (1, 4096, 1024, 4096), (129, 4096, 1024, 4096)],
    )
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    @parametrize("inplace", [False, True])
    def test_qkv_int4(
        self, m, n_q, n_kv, k, per_channel, qmode: QuantMode, dtype, inplace
    ):
        if (dtype == torch.bfloat16) and skip_bf16_input:
            pytest.skip("bf16 input is not available on mtl")
        elif (dtype == torch.bfloat16) and (COMPILER_VERSION < 20240200):
            pytest.skip("bf16 input is only available on OneAPI 2024.2 and above")
        group_size = min(32, k)
        if per_channel:
            group_size = k

        input = torch.rand([m, k], device="xpu", dtype=dtype)
        bias = [torch.rand([n], device="xpu", dtype=dtype) for n in (n_q, n_kv, n_kv)]
        weight = [
            torch.randint(0, 1112111, [k // 8, n], device="xpu").to(torch.int32)
            for n in (n_q, n_kv, n_kv)
        ]
        checking_atol = base_atol
        checking_rtol = base_rtol

        group_num = k // group_size

        scales = [
            torch.rand([group_num, n], device="xpu", dtype=dtype)
            for n in (n_q, n_kv, n_kv)
        ]
        if qmode == QuantMode.SYM:
            zero_points = (None,) * 3
        elif qmode == QuantMode.ASYM:
            zero_points = [
                self.rand_int4(group_num * n, torch.int32, "xpu").reshape(
                    group_num, n // 8
                )
                for n in (n_q, n_kv, n_kv)
            ]
        elif qmode == QuantMode.ASYM_FP_ZP:
            zero_points = [
                torch.rand([group_num, n], device="xpu", dtype=dtype)
                for n in (n_q, n_kv, n_kv)
            ]

        weight_fp16 = [
            self.dequantize(weight[i], scales[i], zero_points[i], group_size).to(dtype)
            for i in range(3)
        ]
        out_torch = [
            torch.matmul(input.cpu().float(), weight_fp16[i].cpu().float())
            for i in range(3)
        ]
        out_torch_bias = [out_torch[i] + bias[i].cpu().float() for i in range(3)]

        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
            if inplace:
                out_xetla = [
                    # use -999 for throughout testing; use torch.empty in real use case
                    torch.full([m, n], -999, device="xpu", dtype=dtype)
                    for n in (n_q, n_kv, n_kv)
                ]
                torch.ops.torch_ipex.mm_qkv_out_int4(
                    input,
                    torch.cat([w.t() for w in weight]).contiguous(),
                    torch.cat([s.t() for s in scales]).contiguous(),
                    (
                        None
                        if qmode == QuantMode.SYM
                        else torch.cat(zero_points, dim=1).contiguous()
                    ),
                    torch.cat(bias).contiguous(),
                    *out_xetla,
                    group_size,
                )
            else:
                out_xetla = torch.ops.torch_ipex.mm_qkv_int4(
                    input,
                    torch.cat([w.t() for w in weight]).contiguous(),
                    torch.cat([s.t() for s in scales]).contiguous(),
                    (
                        None
                        if qmode == QuantMode.SYM
                        else torch.cat(zero_points, dim=1).contiguous()
                    ),
                    torch.cat(bias).contiguous(),
                    n_kv,
                    n_kv,
                    group_size,
                )
        self.assertEqual(
            torch.cat([out_xetla[i].cpu().float() for i in range(3)], dim=-1),
            torch.cat(out_torch_bias, dim=-1),
            atol=checking_atol,
            rtol=checking_rtol,
        )

    @parametrize("per_channel", [False], lambda k: "per_channel" * k)
    @parametrize("dtype", [torch.float16])
    @parametrize("act_order", [False, True])
    @parametrize("m,n,k", [(8, 4096, 4096), (1, 4096, 11008), (32, 4096, 4096)])
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_gptq_woqlinear_interface(
        self, m, n, k, per_channel, act_order, dtype=torch.float16
    ):
        input = torch.rand([m, k], device="xpu", dtype=dtype)
        input_torch = input.cpu()
        weight = self.rand_int4(k * n, torch.int32, "xpu").reshape(k // 8, n)
        group_size = min(128, k)
        checking_atol = base_atol
        checking_rtol = base_rtol
        g_idx = None
        if act_order:
            g_idx = torch.randperm(k, dtype=torch.int32) // group_size
        if per_channel:
            group_size = k
        group_num = int(k / group_size)

        scales = -torch.rand([group_num, n], device="xpu", dtype=dtype)
        zero_points = None
        zero_points_kernel = None

        weight_fp16 = self.dequantize(
            weight, scales, zero_points, group_size, g_idx
        ).cpu()

        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
            woqlinear = ipex.llm.quantization.IPEXWeightOnlyQuantizedLinear.from_weight(
                weight,
                scales,
                zero_points_kernel,
                k,
                n,
                None,
                None,
                group_size,
                g_idx.to("xpu") if g_idx is not None else None,
                ipex.llm.quantization.QuantMethod.GPTQ_GEMM,
                ipex.llm.quantization.QuantDtype.INT4,
            )
            out_xetla = woqlinear(input)
        out_torch = torch.matmul(input_torch, weight_fp16)
        self.assertEqual(
            out_xetla.cpu().float(),
            out_torch.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

    @parametrize("per_channel", [False], lambda k: "per_channel" * k)
    @parametrize("dtype", [torch.float16])
    @parametrize("act_order", [False])
    @parametrize("m,n,k", [(8, 4096, 4096), (1, 4096, 11008), (32, 4096, 4096)])
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_awq_woqlinear_interface(
        self, m, n, k, per_channel, act_order, dtype=torch.float16
    ):
        input = torch.rand([m, k], device="xpu", dtype=dtype)
        input_torch = input.cpu()
        weight = self.rand_int4(k * n, torch.int32, "xpu").reshape(k, n // 8)
        group_size = min(128, k)
        checking_atol = 2e-1
        checking_rtol = base_rtol
        g_idx = None
        g_idx4kernel = None
        if per_channel:
            group_size = k
        group_num = int(k / group_size)

        scales = -torch.rand([group_num, n], device="xpu", dtype=dtype)
        zero_points = self.rand_int4(group_num * n, torch.int32, "xpu").reshape(
            group_num, n // 8
        )

        weight_gptq, zero_points_gptq = fast_awq_to_gptq(weight, zero_points)

        weight_fp16 = self.dequantize(
            weight_gptq, scales, zero_points_gptq, group_size, g_idx
        ).cpu()
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
            woqlinear = ipex.llm.quantization.IPEXWeightOnlyQuantizedLinear.from_weight(
                weight,
                scales,
                zero_points,
                k,
                n,
                None,
                None,
                group_size,
                g_idx4kernel,
                ipex.llm.quantization.QuantMethod.AWQ_GEMM,
                ipex.llm.quantization.QuantDtype.INT4,
            )
            out_xetla = woqlinear(input)
        out_torch = torch.matmul(input_torch, weight_fp16)
        print("out_xetla: ", out_xetla.cpu().float())
        print("out_torch: ", out_torch.float())
        self.assertEqual(
            out_xetla.cpu().float(),
            out_torch.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

    @parametrize("per_channel", [False], lambda k: "per_channel" * k)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("qmode", list(QuantMode), lambda k: k.name)
    @parametrize("act_order", [False, True])
    @parametrize("m,n,k", [(8, 4096, 4096), (1, 4096, 11008), (32, 4096, 4096)])
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_gemm_int4(self, m, n, k, per_channel, act_order, qmode: QuantMode, dtype):
        if (dtype == torch.bfloat16) and skip_bf16_input:
            pytest.skip("bf16 input is not available on mtl")
        elif (dtype == torch.bfloat16) and (COMPILER_VERSION < 20240200):
            pytest.skip("bf16 input is only available on OneAPI 2024.2 and above")

        checking_atol = base_atol
        checking_rtol = base_rtol
        if qmode == QuantMode.ASYM:  # sym needs more tolerance
            checking_atol = 2e-1
            checking_rtol = 5e-2
        input = torch.rand([m, k], device="xpu", dtype=dtype)
        input_torch = input.cpu()
        weight = self.rand_int4(k * n, torch.int32, "xpu").reshape(k // 8, n)

        group_size = min(128, k)
        if per_channel:
            group_size = k
        group_num = k // group_size

        scales = -torch.rand([group_num, n], device="xpu", dtype=dtype)
        if qmode == QuantMode.SYM:
            zero_points = None
        elif qmode == QuantMode.ASYM:
            zero_points = self.rand_int4(group_num * n, torch.int32, "xpu").reshape(
                group_num, n // 8
            )
        elif qmode == QuantMode.ASYM_FP_ZP:
            zero_points = torch.rand([group_num, n], device="xpu", dtype=dtype)

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
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
            out_xetla = torch.ops.torch_ipex.mm_int4(
                input,
                shuffled_weight.t().contiguous(),
                scales.t().contiguous(),
                zero_points,
                group_size,
                g_idx4kernel,
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
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
            out_xetla_res = torch.ops.torch_ipex.mm_add_int4(
                input,
                shuffled_weight.t().contiguous(),
                scales.t().contiguous(),
                zero_points,
                group_size,
                res0,
                g_idx4kernel,
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
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
            out_xetla_bias = torch.ops.torch_ipex.mm_bias_int4(
                input,
                shuffled_weight.t().contiguous(),
                bias,
                scales.t().contiguous(),
                zero_points,
                group_size,
                g_idx4kernel,
            )
        out_torch_bias = out_torch + bias.cpu().float()
        self.assertEqual(
            out_xetla_bias.cpu().float(),
            out_torch_bias.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias + gelu
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
            out_xetla_gelu = torch.ops.torch_ipex.mm_bias_gelu_int4(
                input,
                shuffled_weight.t().contiguous(),
                scales.t().contiguous(),
                zero_points,
                bias,
                group_size,
                "tanh",
                g_idx4kernel,
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
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
            out_xetla_silu = torch.ops.torch_ipex.mm_silu_mul_int4(
                input,
                shuffled_weight.t().contiguous(),
                scales.t().contiguous(),
                zero_points,
                group_size,
                res0,
                g_idx4kernel,
            )
        silu_mul_out = torch.nn.SiLU()(out_torch) * res0.cpu().float()
        self.assertEqual(
            out_xetla_silu.cpu().float(),
            silu_mul_out.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias + silu + mul
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
            out_xetla_silu = torch.ops.torch_ipex.mm_bias_silu_mul_int4(
                input,
                shuffled_weight.t().contiguous(),
                bias,
                scales.t().contiguous(),
                zero_points,
                group_size,
                res0,
                g_idx4kernel,
            )
        silu_mul_out = torch.nn.SiLU()(out_torch_bias) * res0.cpu().float()
        self.assertEqual(
            out_xetla_silu.cpu().float(),
            silu_mul_out.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # check gemm + bias + residual + residual
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        res1 = torch.rand([m, n], device="xpu", dtype=dtype)
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
            out_xetla_bias_2res = torch.ops.torch_ipex.mm_bias_resadd_resadd_int4(
                input,
                shuffled_weight.t().contiguous(),
                bias,
                res0,
                res1,
                scales.t().contiguous(),
                zero_points,
                group_size,
                g_idx4kernel,
            )
        out_torch_bias_2res = out_torch_bias + res0.cpu().float() + res1.cpu().float()
        self.assertEqual(
            out_xetla_bias_2res.cpu().float(),
            out_torch_bias_2res.float(),
            atol=checking_atol,
            rtol=checking_rtol * 2,
        )

        # check gemm + bias + residual
        res0 = torch.rand([m, n], device="xpu", dtype=dtype)
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
            out_xetla_bias_add = torch.ops.torch_ipex.mm_bias_add_int4(
                input,
                shuffled_weight.t().contiguous(),
                bias,
                scales.t().contiguous(),
                zero_points,
                group_size,
                res0,
                g_idx4kernel,
            )
        out_torch_bias_add = out_torch_bias + res0.cpu().float()
        self.assertEqual(
            out_xetla_bias_add.cpu().float(),
            out_torch_bias_add.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

    @parametrize("per_channel", [False], lambda k: "per_channel" * k)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("qmode", list(QuantMode), lambda k: k.name)
    @parametrize("m,n,k", [(8, 4096, 4096), (1, 4096, 11008), (32, 4096, 4096)])
    @pytest.mark.skipif(not torch.xpu.has_xetla(), reason="fallback is required")
    def test_mlp_int4(self, m, n, k, per_channel, qmode: QuantMode, dtype):
        if (dtype == torch.bfloat16) and skip_bf16_input:
            pytest.skip("bf16 input is not available on mtl")
        elif (dtype == torch.bfloat16) and (COMPILER_VERSION < 20240200):
            pytest.skip("bf16 input is only available on OneAPI 2024.2 and above")
        checking_atol = 10  # elt mul will introduce more error
        checking_rtol = base_rtol * 5
        if dtype == torch.bfloat16:
            checking_atol *= 5  # bf16 has larger error
        input = torch.rand([m, k], device="xpu", dtype=dtype) - 0.5
        input_torch = input.cpu()
        weight = self.rand_int4(k * n, torch.int32, "xpu").reshape(k // 8, n)

        group_size = min(128, k)
        if per_channel:
            group_size = k
        group_num = k // group_size

        scales = torch.rand([group_num, n], device="xpu", dtype=dtype)
        if qmode == QuantMode.SYM:
            zero_points = None
        elif qmode == QuantMode.ASYM:
            zero_points = self.rand_int4(group_num * n, torch.int32, "xpu").reshape(
                group_num, n // 8
            )
        elif qmode == QuantMode.ASYM_FP_ZP:
            zero_points = torch.rand([group_num, n], device="xpu", dtype=dtype)

        bias = torch.rand([1, n], device="xpu", dtype=dtype) - 0.5

        weight_fp = self.dequantize(weight, scales, zero_points, group_size).cpu()
        out_torch = torch.matmul(input_torch, weight_fp)
        out_torch_bias = out_torch + bias.cpu()

        # mlp silu mul
        weight_up = self.rand_int4(k * n, torch.int32, "xpu").reshape(k // 8, n)
        scales_up = torch.rand([group_num, n], device="xpu", dtype=dtype)
        if qmode == QuantMode.SYM:
            zero_points_up = None
        elif qmode == QuantMode.ASYM:
            zero_points_up = self.rand_int4(group_num * n, torch.int32, "xpu").reshape(
                group_num, n // 8
            )
        elif qmode == QuantMode.ASYM_FP_ZP:
            zero_points_up = torch.rand([group_num, n], device="xpu", dtype=dtype)
        weight_up_fp = self.dequantize(
            weight_up, scales_up, zero_points_up, group_size
        ).cpu()
        out_torch_silu = torch.nn.SiLU()(out_torch)
        out_torch_up = torch.matmul(input_torch, weight_up_fp)
        out_torch_mlp_silu_mul = out_torch_silu * out_torch_up
        xetla_mlp_args_common = (
            input,
            torch.stack((weight.t(), weight_up.t())).contiguous(),
            torch.stack((scales.t(), scales_up.t())).contiguous(),
            (
                None
                if zero_points is None
                else torch.stack((zero_points, zero_points_up)).contiguous()
            ),
        )
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
            out_xetla_mlp_silu_mul = torch.ops.torch_ipex.mlp_silu_mul_int4(
                *xetla_mlp_args_common,
                group_size,
            )
        self.assertEqual(
            out_xetla_mlp_silu_mul.cpu().float(),
            out_torch_mlp_silu_mul.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # mlp bias silu mul
        out_torch_bias_silu = torch.nn.SiLU()(out_torch_bias)
        out_torch_mlp_bias_silu_mul = out_torch_bias_silu * out_torch_up
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
            out_xetla_mlp_bias_silu_mul = torch.ops.torch_ipex.mlp_bias_silu_mul_int4(
                *xetla_mlp_args_common,
                bias,
                group_size,
            )
        self.assertEqual(
            out_xetla_mlp_bias_silu_mul.cpu().float(),
            out_torch_mlp_bias_silu_mul.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # mlp silu mul bias
        bias_up = torch.rand([1, n], device="xpu", dtype=dtype) - 0.5
        out_torch_bias_up = out_torch_up + bias_up.cpu()
        out_torch_mlp_silu_mul_bias = out_torch_silu * out_torch_bias_up
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
            out_xetla_mlp_silu_mul_bias = torch.ops.torch_ipex.mlp_silu_mul_bias_int4(
                *xetla_mlp_args_common,
                bias_up,
                group_size,
            )
        self.assertEqual(
            out_xetla_mlp_silu_mul_bias.cpu().float(),
            out_torch_mlp_silu_mul_bias.float(),
            atol=checking_atol,
            rtol=checking_rtol,
        )

        # mlp bias silu mul bias
        out_torch_mlp_bias_silu_mul_bias = out_torch_bias_silu * out_torch_bias_up
        with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
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
            atol=checking_atol,
            rtol=checking_rtol,
        )


instantiate_parametrized_tests(TestInt4Linear)

if __name__ == "__main__":
    run_tests()
