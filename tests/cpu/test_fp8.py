import torch
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import run_tests, TestCase
import time


class TestFloat8(TestCase):
    def test_creation_with_zeros(self):
        x = torch.zeros(8, dtype=torch.float8_e4m3fn)

    def test_e4m3fn_casts(self):
        for dtype in (torch.float32, torch.float16):
            x = torch.randn(16, dtype=torch.float)
            x_fp8 = x.to(torch.float8_e4m3fn)
            x_orig_dtype = x_fp8.to(torch.float)

    def test_e4m3fn_numerics(self):
        # ensure that our format matches https://arxiv.org/pdf/2209.05433.pdf, Table 1

        def _compare(bits_str, expected_fp32, comp_name):
            bits_int = int(bits_str, 2)
            tensor_int = torch.tensor([bits_int], dtype=torch.uint8)
            tensor_fp8 = tensor_int.view(torch.float8_e4m3fn)
            tensor_fp32 = tensor_fp8.float()
            ref_tensor_fp32 = torch.tensor([expected_fp32], dtype=torch.float)
            self.assertTrue(
                torch.allclose(tensor_fp32, ref_tensor_fp32),
                f"{comp_name} failed: expected {expected_fp32}, got {tensor_fp32.item()}",
            )

        _compare("00000000", 0.0, "zero")
        _compare("10000000", -0.0, "neg_zero")
        _compare("01111110", 448.0, "max_normal")
        _compare("11111110", -448.0, "neg_max_normal")
        _compare("00001000", 2**-6, "min_normal")
        _compare("10001000", -1 * (2**-6), "neg_min_normal")
        _compare("00000111", 0.875 * (2**-6), "max_subnorm")
        _compare("10000111", -0.875 * (2**-6), "neg_max_subnorm")
        _compare("00000001", 2**-9, "min_subnorm")
        _compare("10000001", -1 * (2**-9), "neg_min_subnorm")

    def test_e5m2fn_casts(self):
        for dtype in (torch.float32, torch.float16):
            x = torch.randn(16, dtype=torch.float)
            x_fp8 = x.to(torch.float8_e5m2)
            x_orig_dtype = x_fp8.to(torch.float)

    def test_e5m2fn_numerics(self):
        # ensure that our format matches https://arxiv.org/pdf/2209.05433.pdf, Table 1

        def _compare(bits_str, expected_fp32, comp_name):
            bits_int = int(bits_str, 2)
            tensor_int = torch.tensor([bits_int], dtype=torch.uint8)
            tensor_fp8 = tensor_int.view(torch.float8_e5m2)
            tensor_fp32 = tensor_fp8.float()
            ref_tensor_fp32 = torch.tensor([expected_fp32], dtype=torch.float)
            self.assertTrue(
                torch.allclose(tensor_fp32, ref_tensor_fp32),
                f"{comp_name} failed: expected {expected_fp32}, got {tensor_fp32.item()}",
            )

        _compare("00000000", 0.0, "zero")
        _compare("10000000", -0.0, "neg_zero")
        _compare("01111011", 57344.0, "max_normal")
        _compare("11111011", -57344.0, "neg_max_normal")
        _compare("00000100", 2**-14, "min_normal")
        _compare("10000100", -1 * (2**-14), "neg_min_normal")
        _compare("00000011", 0.75 * (2**-14), "max_subnorm")
        _compare("10000011", -0.75 * (2**-14), "neg_max_subnorm")
        _compare("00000001", 2**-16, "min_subnorm")
        _compare("10000001", -1 * (2**-16), "neg_min_subnorm")

    def test_convert_e4m3_to_bf16(self):
        # Test without denorm
        weight = (torch.rand(4096, 14336, dtype=torch.bfloat16) * 2 - 1) * 448.0
        weight = weight.to(torch.float8_e4m3fn)
        weight_bf16 = torch.empty(4096, 14336, dtype=torch.bfloat16)
        t0 = time.time()
        for i in range(50):
            torch.ops.torch_ipex.convert_e4m3_to_bf16(
                weight, weight_bf16, 4096 * 14336, False, False
            )
        print(
            "The time of convert_e4m3_to_bf16_without_denorm is",
            (time.time() - t0) / 50,
            "s",
        )
        self.assertTrue(
            torch.allclose(weight.to(torch.bfloat16), weight_bf16),
            f"convert_e4m3_to_bf16 failed: expected {weight}, got {weight_bf16}",
        )

        # Test with denorm using intrinsic (No Nan support)
        weight = torch.arange(256, dtype=torch.uint8)
        weight[127] = 126
        weight[255] = 254
        weight = weight.view(torch.float8_e4m3fn)
        weight_bf16 = torch.empty(256, dtype=torch.bfloat16)
        torch.ops.torch_ipex.convert_e4m3_to_bf16(weight, weight_bf16, 256, True, False)
        self.assertTrue(
            torch.allclose(weight.to(torch.bfloat16), weight_bf16, equal_nan=True),
            f"convert_e4m3_to_bf16 failed: expected {weight}, got {weight_bf16}",
        )
        weight2 = torch.randn(4096, 14336, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        weight2_bf16 = torch.empty(4096, 14336, dtype=torch.bfloat16)
        t0 = time.time()
        for i in range(50):
            torch.ops.torch_ipex.convert_e4m3_to_bf16(
                weight2, weight2_bf16, 4096 * 14336, True, False
            )
        print(
            "The time of convert_e4m3_to_bf16_with_denorm using intrinsic is",
            (time.time() - t0) / 50,
            "s",
        )
        self.assertTrue(
            torch.allclose(weight2.to(torch.bfloat16), weight2_bf16),
            f"convert_e4m3_to_bf16 failed: expected {weight2}, got {weight2_bf16}",
        )

        # Test with denorm using lut
        weight_bf16 = torch.empty(256, dtype=torch.bfloat16)
        torch.ops.torch_ipex.convert_e4m3_to_bf16(weight, weight_bf16, 256, True, True)
        self.assertTrue(
            torch.allclose(weight.to(torch.bfloat16), weight_bf16, equal_nan=True),
            f"convert_e4m3_to_bf16 failed: expected {weight}, got {weight_bf16}",
        )
        weight2_bf16 = torch.empty(4096, 14336, dtype=torch.bfloat16)
        t0 = time.time()
        for i in range(50):
            torch.ops.torch_ipex.convert_e4m3_to_bf16(
                weight2, weight2_bf16, 4096 * 14336, True, False
            )
        print(
            "The time of convert_e4m3_to_bf16_with_denorm using lut is",
            (time.time() - t0) / 50,
            "s",
        )
        self.assertTrue(
            torch.allclose(weight2.to(torch.bfloat16), weight2_bf16),
            f"convert_e4m3_to_bf16 failed: expected {weight2}, got {weight2_bf16}",
        )

    def test_convert_e4m3_to_fp32(self):
        # Test without denorm
        weight = (torch.rand(4096, 14336, dtype=torch.bfloat16) * 2 - 1) * 448.0
        weight = weight.to(torch.float8_e4m3fn)
        weight_fp32 = torch.empty(4096, 14336, dtype=torch.float32)
        t0 = time.time()
        for i in range(50):
            torch.ops.torch_ipex.convert_e4m3_to_fp32(
                weight, weight_fp32, 4096 * 14336, False, False
            )
        print(
            "The time of convert_e4m3_to_fp32_without_denorm is",
            (time.time() - t0) / 50,
            "s",
        )
        self.assertTrue(
            torch.allclose(weight.to(torch.float32), weight_fp32),
            f"convert_e4m3_to_fp32 failed: expected {weight}, got {weight_fp32}",
        )

        # Test with denorm using lut
        weight = torch.arange(256, dtype=torch.uint8)
        weight = weight.view(torch.float8_e4m3fn)
        weight_fp32 = torch.empty(256, dtype=torch.float32)
        torch.ops.torch_ipex.convert_e4m3_to_fp32(weight, weight_fp32, 256, True, True)
        self.assertTrue(
            torch.allclose(weight.to(torch.float32), weight_fp32, equal_nan=True),
            f"convert_e4m3_to_fp32 failed: expected {weight}, got {weight_fp32}",
        )
        weight2 = torch.randn(4096, 14336, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        weight2_fp32 = torch.empty(4096, 14336, dtype=torch.float32)
        t0 = time.time()
        for i in range(50):
            torch.ops.torch_ipex.convert_e4m3_to_fp32(
                weight2, weight2_fp32, 4096 * 14336, True, True
            )
        print(
            "The time of convert_e4m3_to_fp32_with_denorm using lut is",
            (time.time() - t0) / 50,
            "s",
        )
        self.assertTrue(
            torch.allclose(weight2.to(torch.float32), weight2_fp32),
            f"convert_e4m3_to_fp32 failed: expected {weight2}, got {weight2_fp32}",
        )

    def test_convert_e4m3_to_lut_fp16(self):
        # Test with denorm using lut
        weight = torch.arange(256, dtype=torch.uint8)
        weight = weight.view(torch.float8_e4m3fn)
        weight_fp16 = torch.empty(256, dtype=torch.float16)
        torch.ops.torch_ipex.convert_e4m3_to_fp16(weight, weight_fp16, 256, True, True)
        self.assertTrue(
            torch.allclose(weight.to(torch.float16), weight_fp16, equal_nan=True),
            f"convert_e4m3_to_fp16 failed: expected {weight}, got {weight_fp16}",
        )

        weight = torch.randn(4096, 14336, dtype=torch.float16).to(torch.float8_e4m3fn)
        weight_fp16 = torch.empty(4096, 14336, dtype=torch.float16)
        t0 = time.time()
        for i in range(50):
            torch.ops.torch_ipex.convert_e4m3_to_fp16(
                weight, weight_fp16, 4096 * 14336, True, True
            )
        print(
            "The time of convert_e4m3_to_fp16_with_denorm using lut is",
            (time.time() - t0) / 50,
            "s",
        )
        self.assertTrue(
            torch.allclose(weight.to(torch.float16), weight_fp16),
            f"convert_e4m3_to_fp16 failed: expected {weight}, got {weight_fp16}",
        )

    def test_convert_e5m2_to_fp16(self):
        weight = torch.randn(4096, 14336, dtype=torch.float16).to(torch.float8_e5m2)
        weight_fp16 = torch.empty(4096, 14336, dtype=torch.float16)
        t0 = time.time()
        for i in range(50):
            torch.ops.torch_ipex.convert_e5m2_to_fp16(weight, weight_fp16, 4096 * 14336)
        print("The time of convert_e5m2_to_fp16 is", (time.time() - t0) / 50, "s")
        self.assertTrue(
            torch.allclose(weight.to(torch.float16), weight_fp16),
            f"convert_e4m3_to_fp16 failed: expected {weight}, got {weight_fp16}",
        )


if __name__ == "__main__":
    run_tests()
