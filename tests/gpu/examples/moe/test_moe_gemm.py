import torch
import pytest

import intel_extension_for_pytorch  # noqa

dpcpp_device = torch.device("xpu")


class TestTorchMethod:

    def init_rows_for_experts(self, tokens, topk, rows_for_experts):
        n_experts = rows_for_experts.numel()
        rand = torch.rand(tokens, n_experts, device=rows_for_experts.device)
        topk_idx = torch.topk(rand, topk, dim=1).indices  # [tokens, topk]
        flat_idx = topk_idx.flatten()
        rows_for_experts += torch.bincount(flat_idx, minlength=n_experts)

    def validata_moe_gemm(
        self, matrix_a, matrix_b, rows_for_experts, rows_for_experts_cpu, n_experts
    ):
        total_m = matrix_a.shape[0]
        gemm_k = matrix_a.shape[1]
        gemm_n = matrix_b.shape[2]

        output = torch.zeros(total_m, gemm_n, device=dpcpp_device)
        start = 0
        for i in range(n_experts):
            end = start + rows_for_experts_cpu[i].item()
            output[start:end] = torch.mm(matrix_a[start:end], matrix_b[i])
            start = end

        return output

    @pytest.mark.parametrize("tokens", [1024])
    @pytest.mark.parametrize("topk", [8])
    @pytest.mark.parametrize("gemm_k", [1024, 4096])
    @pytest.mark.parametrize("gemm_n", [1024, 16384])
    @pytest.mark.parametrize("n_experts", [8, 16, 32, 64])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_moe_gemm(self, n_experts, gemm_k, gemm_n, tokens, topk, dtype):

        total_m = tokens * topk
        matrix_a = torch.randn(total_m, gemm_k, dtype=dtype, device=dpcpp_device)
        matrix_b = torch.randn(
            n_experts, gemm_k, gemm_n, dtype=dtype, device=dpcpp_device
        )

        rows_for_experts = torch.zeros(
            n_experts, device=dpcpp_device, dtype=torch.int32
        )
        self.init_rows_for_experts(tokens, topk, rows_for_experts)

        output = torch.xpu.moe_gemm(matrix_a, matrix_b, rows_for_experts, n_experts)
        rows_for_experts_cpu = rows_for_experts.to(torch.int32).to("cpu")
        ref_output = self.validata_moe_gemm(
            matrix_a, matrix_b, rows_for_experts, rows_for_experts_cpu, n_experts
        )
        torch.testing.assert_close(
            output.to(float), ref_output.to(float), rtol=1e-2, atol=1e-2
        )

    @pytest.mark.parametrize("tokens", [1024])
    @pytest.mark.parametrize("topk", [8])
    @pytest.mark.parametrize("gemm_k", [1024, 4096])
    @pytest.mark.parametrize("gemm_n", [1024, 16384])
    @pytest.mark.parametrize("n_experts", [8, 16, 32, 64])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e5m2])
    def test_moe_gemm_fp8(
        self, n_experts, gemm_k, gemm_n, tokens, topk, dtype, fp8_dtype
    ):

        total_m = tokens * topk
        matrix_a = torch.randn(total_m, gemm_k, dtype=dtype, device=dpcpp_device)
        matrix_b = torch.randn(
            n_experts, gemm_k, gemm_n, dtype=dtype, device=dpcpp_device
        )

        rows_for_experts = torch.zeros(
            n_experts, device=dpcpp_device, dtype=torch.int32
        )
        self.init_rows_for_experts(tokens, topk, rows_for_experts)

        scale_shape = None
        matrix_b_scale = torch.full((n_experts,), 4.0, device=dpcpp_device)
        matrix_b_scale_inv = torch.full((n_experts,), 0.25, device=dpcpp_device)
        matrix_b_fp8 = torch.empty_like(matrix_b, device=dpcpp_device, dtype=fp8_dtype)
        matrix_b_dequant = torch.empty_like(matrix_b, device=dpcpp_device)
        for i in range(n_experts):
            matrix_b_fp8[i], _ = torch.ops.torch_ipex.cast_to_fp8(
                matrix_b[i], matrix_b_scale[i], False, False, fp8_dtype, scale_shape
            )
            matrix_b_dequant[i] = torch.ops.torch_ipex.cast_from_fp8(
                matrix_b_fp8[i], matrix_b_scale_inv[i], dtype
            )
        output_fp8 = torch.xpu.moe_gemm(
            matrix_a,
            matrix_b_fp8,
            rows_for_experts,
            n_experts,
            None,
            matrix_b_scale_inv,
        )
        rows_for_experts_cpu = rows_for_experts.to(torch.int32).to("cpu")
        ref_output = self.validata_moe_gemm(
            matrix_a,
            matrix_b_dequant,
            rows_for_experts,
            rows_for_experts_cpu,
            n_experts,
        )
        torch.testing.assert_close(
            output_fp8.to(float),
            ref_output.to(float),
            rtol=1e-2,
            atol=1e-2,
            equal_nan=True,
        )
