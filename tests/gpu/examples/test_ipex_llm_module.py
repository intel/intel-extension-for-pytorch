import copy
import torch

from torch.testing._internal.common_utils import TestCase
from intel_extension_for_pytorch.llm.modules import (
    LinearAdd,
    LinearAddAdd,
    LinearGelu,
    Linear2SiluMul,
)


class TestLLMModules(TestCase):

    def test_LinearAdd(self):
        for dtype in [torch.float, torch.bfloat16, torch.float16]:
            if dtype == torch.float:
                atol = 1e-3
                rtol = 1e-6
            elif dtype == torch.bfloat16:
                atol = 1e-2
                rtol = 1e-2
            elif dtype == torch.float16:
                atol = 1e-3
                rtol = 1e-3

            input_1 = torch.randn(4096, 4096).to(dtype)
            input_2 = torch.randn(4096, 4096).to(dtype)
            input_1_gpu = input_1.clone().to("xpu")
            input_2_gpu = input_2.clone().to("xpu")

            linear_module = torch.nn.Linear(4096, 4096).to(dtype)
            linear_module_gpu = copy.deepcopy(linear_module)

            linear_module_no_bias = copy.deepcopy(linear_module)
            linear_module_no_bias.bias = None

            linear_module_no_bias_gpu = copy.deepcopy(linear_module)
            linear_module_no_bias_gpu.bias = None

            linear_add_fusion = LinearAdd(linear_module)
            linear_add_fusion_gpu = LinearAdd(linear_module_gpu).to("xpu")

            result = linear_add_fusion(input_1, input_2)
            result_gpu = linear_add_fusion_gpu(input_1_gpu, input_2_gpu)
            self.assertEqual(result, result_gpu.cpu(), atol=atol, rtol=rtol)

            linear_add_fusion_no_bias = LinearAdd(linear_module_no_bias)
            linear_add_fusion_no_bias_gpu = LinearAdd(linear_module_no_bias_gpu).to(
                "xpu"
            )

            result = linear_add_fusion_no_bias(input_1, input_2)
            result_gpu = linear_add_fusion_no_bias_gpu(input_1_gpu, input_2_gpu)
            self.assertEqual(result, result_gpu.cpu(), atol=atol, rtol=rtol)

    def test_LinearAddAdd(self):
        for dtype in [torch.float, torch.bfloat16, torch.float16]:
            if dtype == torch.float:
                atol = 1e-3
                rtol = 1e-6
            elif dtype == torch.bfloat16:
                atol = 1e-1
                rtol = 1e-2
            elif dtype == torch.float16:
                atol = 1e-2
                rtol = 1e-3

            input_1 = torch.randn(4096, 4096).to(dtype)
            input_2 = torch.randn(4096, 4096).to(dtype)
            input_3 = torch.randn(4096, 4096).to(dtype)
            input_1_gpu = input_1.clone().to("xpu")
            input_2_gpu = input_2.clone().to("xpu")
            input_3_gpu = input_3.clone().to("xpu")

            linear_module = torch.nn.Linear(4096, 4096).to(dtype)
            linear_module_gpu = copy.deepcopy(linear_module)

            linear_module_no_bias = copy.deepcopy(linear_module)
            linear_module_no_bias.bias = None

            linear_module_no_bias_gpu = copy.deepcopy(linear_module)
            linear_module_no_bias_gpu.bias = None

            linear_add_add_fusion = LinearAddAdd(linear_module)
            linear_add_add_fusion_gpu = LinearAddAdd(linear_module_gpu).to("xpu")

            result = linear_add_add_fusion(input_1, input_2, input_3)
            result_gpu = linear_add_add_fusion_gpu(
                input_1_gpu, input_2_gpu, input_3_gpu
            )
            self.assertEqual(result, result_gpu.cpu(), atol=atol, rtol=rtol)

            linear_add_fusion_no_bias = LinearAddAdd(linear_module_no_bias)
            linear_add_fusion_no_bias_gpu = LinearAddAdd(linear_module_no_bias_gpu).to(
                "xpu"
            )

            result = linear_add_fusion_no_bias(input_1, input_2, input_3)
            result_gpu = linear_add_fusion_no_bias_gpu(
                input_1_gpu, input_2_gpu, input_3_gpu
            )
            self.assertEqual(result, result_gpu.cpu(), atol=atol, rtol=rtol)

    def test_LinearGelu(self):
        for dtype in [torch.float, torch.bfloat16, torch.float16]:
            if dtype == torch.float:
                atol = 1e-3
                rtol = 1e-6
            elif dtype == torch.bfloat16:
                atol = 1e-2
                rtol = 1e-2
            elif dtype == torch.float16:
                atol = 1e-3
                rtol = 1e-3

            input_1 = torch.randn(4096, 4096).to(dtype)
            input_1_gpu = input_1.clone().to("xpu")

            linear_module = torch.nn.Linear(4096, 4096).to(dtype)
            linear_module_gpu = copy.deepcopy(linear_module)

            linear_module_no_bias = copy.deepcopy(linear_module)
            linear_module_no_bias.bias = None

            linear_module_no_bias_gpu = copy.deepcopy(linear_module)
            linear_module_no_bias_gpu.bias = None

            linear_Gelu_fusion = LinearGelu(linear_module)
            linear_Gelu_fusion_gpu = LinearGelu(linear_module_gpu).to("xpu")

            result = linear_Gelu_fusion(input_1)
            result_gpu = linear_Gelu_fusion_gpu(input_1_gpu)
            self.assertEqual(result, result_gpu.cpu(), atol=atol, rtol=rtol)

            linear_Gelu_fusion_no_bias = LinearGelu(linear_module_no_bias)
            linear_Gelu_fusion_no_bias_gpu = LinearGelu(linear_module_no_bias_gpu).to(
                "xpu"
            )

            result = linear_Gelu_fusion_no_bias(input_1)
            result_gpu = linear_Gelu_fusion_no_bias_gpu(input_1_gpu)
            self.assertEqual(result, result_gpu.cpu(), atol=atol, rtol=rtol)

    def test_Linear2SiluMul(self):
        for dtype in [torch.float, torch.bfloat16, torch.float16]:
            if dtype == torch.float:
                atol = 1e-3
                rtol = 1e-6
            elif dtype == torch.bfloat16:
                atol = 1e-1
                rtol = 1e-2
            elif dtype == torch.float16:
                atol = 1e-2
                rtol = 1e-3

            input_1 = torch.randn(4096, 4096).to(dtype)
            input_1_gpu = input_1.clone().to("xpu")

            linear_module = torch.nn.Linear(4096, 4096).to(dtype)
            linear_module_2 = torch.nn.Linear(4096, 4096).to(dtype)
            linear_module_gpu = copy.deepcopy(linear_module)
            linear_module_2_gpu = copy.deepcopy(linear_module_2)

            linear_module_no_bias = copy.deepcopy(linear_module)
            linear_module_no_bias.bias = None

            linear_module_no_bias_gpu = copy.deepcopy(linear_module)
            linear_module_no_bias_gpu.bias = None

            linear_module_2_no_bias = copy.deepcopy(linear_module_2)
            linear_module_2_no_bias.bias = None

            linear_module_2_no_bias_gpu = copy.deepcopy(linear_module_2)
            linear_module_2_no_bias_gpu.bias = None

            linear_Silu_Mul_fusion = Linear2SiluMul(linear_module, linear_module_2)
            linear_Silu_Mul_fusion_gpu = Linear2SiluMul(
                linear_module_gpu, linear_module_2_gpu
            ).to("xpu")

            result = linear_Silu_Mul_fusion(input_1)
            result_gpu = linear_Silu_Mul_fusion_gpu(input_1_gpu)
            self.assertEqual(result, result_gpu.cpu(), atol=atol, rtol=rtol)

            linear_Silu_Mul_fusion_no_bias = Linear2SiluMul(
                linear_module_no_bias, linear_module_2_no_bias
            )
            linear_Silu_Mul_fusion_no_bias_gpu = Linear2SiluMul(
                linear_module_no_bias_gpu, linear_module_2_no_bias_gpu
            ).to("xpu")

            result = linear_Silu_Mul_fusion_no_bias(input_1)
            result_gpu = linear_Silu_Mul_fusion_no_bias_gpu(input_1_gpu)
            self.assertEqual(result, result_gpu.cpu(), atol=atol, rtol=rtol)
