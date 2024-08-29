import torch
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase
from intel_extension_for_pytorch.quantization.fp8 import (
    fp8_autocast,
    DelayedScaling,
    Format,
    FP8Linear,
)  # noqa


class TestFP8GEMM(TestCase):
    def test_fp8_linear_E5M2(self, dtype=torch.float):
        input = torch.ones(
            [8, 2], requires_grad=True, dtype=dtype, device=torch.device("xpu")
        )
        input_ref = torch.ones(
            [8, 2], requires_grad=True, dtype=dtype, device=torch.device("xpu")
        )
        grad_out = torch.rand([8, 3], dtype=dtype).xpu()
        grad_out_ref = grad_out.clone()

        gemm_ref = torch.nn.Linear(2, 3).xpu()
        gemm_ref.weight.data = torch.rand([3, 2], dtype=dtype).xpu()
        output_ref = gemm_ref(input_ref)
        tanh = torch.nn.Tanh().xpu()

        output2_ref = tanh(output_ref)
        output2_ref.backward(grad_out_ref)
        gd_ref = input_ref.grad
        gw_ref = gemm_ref.weight.grad

        with fp8_autocast(
            enabled=True, fp8_recipe=DelayedScaling(fp8_format=Format.E5M2)
        ):
            gemm = FP8Linear(2, 3).xpu()
            gemm.weight.data = gemm_ref.weight.data.clone()
            gemm.bias.data = gemm_ref.bias.data.clone()
            output = gemm(input)

            output2 = tanh(output)
            output2.backward(grad_out)
            gd = input.grad.clone()
            gw = gemm.weight.grad.clone()

        # print("output_fp8 = ", output)
        # print("output_ref = ", output_ref)
        # print("gd = ", gd)
        # print("gd_ref = ", gd_ref)
        # print("gw = ", gw)
        # print("gw_ref = ", gw_ref)
        self.assertEqual(output, output_ref, rtol=1e-1, atol=1e-2)
        self.assertEqual(gd, gd_ref, rtol=1e-1, atol=1e-2)
        self.assertEqual(gw, gw_ref, rtol=1e-1, atol=1e-1)

    def test_fp8_linear_E4M3(self, dtype=torch.float):
        input = torch.ones(
            [8, 2], requires_grad=True, dtype=dtype, device=torch.device("xpu")
        )
        input_ref = torch.ones(
            [8, 2], requires_grad=True, dtype=dtype, device=torch.device("xpu")
        )
        grad_out = torch.rand([8, 3], dtype=dtype).xpu()
        grad_out_ref = grad_out.clone()

        gemm_ref = torch.nn.Linear(2, 3).xpu()
        gemm_ref.weight.data = torch.rand([3, 2], dtype=dtype).xpu()
        output_ref = gemm_ref(input_ref)
        tanh = torch.nn.Tanh().xpu()

        output2_ref = tanh(output_ref)
        output2_ref.backward(grad_out_ref)
        gd_ref = input_ref.grad
        gw_ref = gemm_ref.weight.grad

        with fp8_autocast(
            enabled=True, fp8_recipe=DelayedScaling(fp8_format=Format.E4M3)
        ):
            gemm = FP8Linear(2, 3).xpu()
            gemm.weight.data = gemm_ref.weight.data.clone()
            gemm.bias.data = gemm_ref.bias.data.clone()
            output = gemm(input)

            output2 = tanh(output)
            output2.backward(grad_out)
            gd = input.grad.clone()
            gw = gemm.weight.grad.clone()

        # print("output_fp8 = ", output)
        # print("output_ref = ", output_ref)
        # print("gd = ", gd)
        # print("gd_ref = ", gd_ref)
        # print("gw = ", gw)
        # print("gw_ref = ", gw_ref)
        self.assertEqual(output, output_ref, rtol=1e-1, atol=1e-2)
        self.assertEqual(gd, gd_ref, rtol=1e-1, atol=1e-2)
        self.assertEqual(gw, gw_ref, rtol=1e-1, atol=1e-2)

    def test_fp8_linear_HYBRID(self, dtype=torch.float):
        input = torch.ones(
            [8, 2], requires_grad=True, dtype=dtype, device=torch.device("xpu")
        )
        input_ref = torch.ones(
            [8, 2], requires_grad=True, dtype=dtype, device=torch.device("xpu")
        )
        grad_out = torch.rand([8, 3], dtype=dtype).xpu()
        grad_out_ref = grad_out.clone()

        gemm_ref = torch.nn.Linear(2, 3).xpu()
        gemm_ref.weight.data = torch.rand([3, 2], dtype=dtype).xpu()
        output_ref = gemm_ref(input_ref)
        tanh = torch.nn.Tanh().xpu()

        output2_ref = tanh(output_ref)
        output2_ref.backward(grad_out_ref)
        gd_ref = input_ref.grad
        gw_ref = gemm_ref.weight.grad

        with fp8_autocast(
            enabled=True, fp8_recipe=DelayedScaling(fp8_format=Format.HYBRID)
        ):
            gemm = FP8Linear(2, 3).xpu()
            gemm.weight.data = gemm_ref.weight.data.clone()
            gemm.bias.data = gemm_ref.bias.data.clone()
            output = gemm(input)

            output2 = tanh(output)
            output2.backward(grad_out)
            gd = input.grad.clone()
            gw = gemm.weight.grad.clone()

        # print("output_fp8 = ", output)
        # print("output_ref = ", output_ref)
        # print("gd = ", gd)
        # print("gd_ref = ", gd_ref)
        # print("gw = ", gw)
        # print("gw_ref = ", gw_ref)
        self.assertEqual(output, output_ref, rtol=1e-1, atol=1e-2)
        self.assertEqual(gd, gd_ref, rtol=1e-1, atol=1e-2)
        self.assertEqual(gw, gw_ref, rtol=1e-1, atol=1e-2)
