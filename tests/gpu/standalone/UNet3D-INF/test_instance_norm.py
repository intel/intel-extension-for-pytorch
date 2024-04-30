import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch as ipex
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestNNMethod(TestCase):
    def test_instance_norm3d_1(self, dtype=torch.float):
        rand_input = torch.randn((1, 128, 56, 56, 40), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(128)
        cpu_result = test_module(rand_input)
        xpu_module = test_module.to("xpu")
        xpu_result = xpu_module(rand_input.to("xpu"))
        self.assertEqual(cpu_result, xpu_result.to("cpu"))

    def test_instance_norm3d_bfloat16_1(self, dtype=torch.bfloat16):
        rand_input = torch.randn((1, 128, 56, 56, 40), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(128)
        cpu_result = test_module(rand_input)
        xpu_module = test_module.to("xpu")
        xpu_result = xpu_module(rand_input.to("xpu"))
        self.assertEqual(cpu_result, xpu_result.to("cpu"), rtol=10e-3, atol=10e-4)

    def test_instance_norm3d_float16_1(self, dtype=torch.bfloat16):
        rand_input = torch.randn((1, 128, 56, 56, 40), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(128)
        cpu_result = test_module(rand_input)
        dtype_dpcpp = torch.float16
        xpu_module = test_module.to("xpu").to(dtype_dpcpp)
        xpu_result = xpu_module(rand_input.to("xpu").to(dtype_dpcpp))
        self.assertEqual(cpu_result, xpu_result.to("cpu").to(torch.bfloat16), rtol=10e-3, atol=10e-4)

    def test_instance_norm3d_2(self, dtype=torch.float):
        rand_input = torch.randn((1, 256, 28, 28, 20), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(256)
        cpu_result = test_module(rand_input)
        xpu_module = test_module.to("xpu")
        xpu_result = xpu_module(rand_input.to("xpu"))
        self.assertEqual(cpu_result, xpu_result.to("cpu"))

    def test_instance_norm3d_bfloat16_2(self, dtype=torch.bfloat16):
        rand_input = torch.randn((1, 256, 28, 28, 20), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(256)
        cpu_result = test_module(rand_input)
        xpu_module = test_module.to("xpu")
        xpu_result = xpu_module(rand_input.to("xpu"))
        self.assertEqual(cpu_result, xpu_result.to("cpu"), rtol=10e-3, atol=10e-4)

    def test_instance_norm3d_float16_2(self, dtype=torch.bfloat16):
        rand_input = torch.randn((1, 256, 28, 28, 20), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(256)
        cpu_result = test_module(rand_input)
        dtype_dpcpp = torch.float16
        xpu_module = test_module.to("xpu").to(dtype_dpcpp)
        xpu_result = xpu_module(rand_input.to("xpu").to(dtype_dpcpp))
        self.assertEqual(cpu_result, xpu_result.to("cpu").to(torch.bfloat16), rtol=10e-3, atol=10e-4)

    def test_instance_norm3d_3(self, dtype=torch.float):
        rand_input = torch.randn((1, 320, 14, 14, 10), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(320)
        cpu_result = test_module(rand_input)
        xpu_module = test_module.to("xpu")
        xpu_result = xpu_module(rand_input.to("xpu"))
        self.assertEqual(cpu_result, xpu_result.to("cpu"))

    def test_instance_norm3d_bfloat16_3(self, dtype=torch.bfloat16):
        rand_input = torch.randn((1, 320, 14, 14, 10), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(320)
        cpu_result = test_module(rand_input)
        xpu_module = test_module.to("xpu")
        xpu_result = xpu_module(rand_input.to("xpu"))
        self.assertEqual(cpu_result, xpu_result.to("cpu"), rtol=10e-3, atol=10e-4)

    def test_instance_norm3d_float16_3(self, dtype=torch.bfloat16):
        rand_input = torch.randn((1, 320, 14, 14, 10), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(320)
        cpu_result = test_module(rand_input)
        dtype_dpcpp = torch.float16
        xpu_module = test_module.to("xpu").to(dtype_dpcpp)
        xpu_result = xpu_module(rand_input.to("xpu").to(dtype_dpcpp))
        self.assertEqual(cpu_result, xpu_result.to("cpu").to(torch.bfloat16), rtol=10e-3, atol=10e-4)

    def test_instance_norm3d_4(self, dtype=torch.float):
        rand_input = torch.randn((1, 320, 7, 7, 5), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(320)
        cpu_result = test_module(rand_input)
        xpu_module = test_module.to("xpu")
        xpu_result = xpu_module(rand_input.to("xpu"))
        self.assertEqual(cpu_result, xpu_result.to("cpu"))

    def test_instance_norm3d_bfloat16_4(self, dtype=torch.bfloat16):
        rand_input = torch.randn((1, 320, 7, 7, 5), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(320)
        cpu_result = test_module(rand_input)
        xpu_module = test_module.to("xpu")
        xpu_result = xpu_module(rand_input.to("xpu"))
        self.assertEqual(cpu_result, xpu_result.to("cpu"), rtol=10e-3, atol=10e-4)

    def test_instance_norm3d_float16_4(self, dtype=torch.bfloat16):
        rand_input = torch.randn((1, 320, 7, 7, 5), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(320)
        cpu_result = test_module(rand_input)
        dtype_dpcpp = torch.float16
        xpu_module = test_module.to("xpu").to(dtype_dpcpp)
        xpu_result = xpu_module(rand_input.to("xpu").to(dtype_dpcpp))
        self.assertEqual(cpu_result, xpu_result.to("cpu").to(torch.bfloat16), rtol=10e-3, atol=10e-3)

    def test_instance_norm3d_5(self, dtype=torch.float):
        rand_input = torch.randn((1, 32, 224, 224, 160), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(32)
        cpu_result = test_module(rand_input)
        xpu_module = test_module.to("xpu")
        xpu_result = xpu_module(rand_input.to("xpu"))
        self.assertEqual(cpu_result, xpu_result.to("cpu"))

    def test_instance_norm3d_bfloat16_5(self, dtype=torch.bfloat16):
        rand_input = torch.randn((1, 32, 224, 224, 160), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(32)
        cpu_result = test_module(rand_input)
        xpu_module = test_module.to("xpu")
        xpu_result = xpu_module(rand_input.to("xpu"))
        self.assertEqual(cpu_result, xpu_result.to("cpu"), rtol=10e-3, atol=10e-4)

    def test_instance_norm3d_float16_5(self, dtype=torch.bfloat16):
        rand_input = torch.randn((1, 32, 224, 224, 160), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(32)
        cpu_result = test_module(rand_input)
        dtype_dpcpp = torch.float16
        xpu_module = test_module.to("xpu").to(dtype_dpcpp)
        xpu_result = xpu_module(rand_input.to("xpu").to(dtype_dpcpp))
        self.assertEqual(cpu_result, xpu_result.to("cpu").to(torch.bfloat16), rtol=10e-3, atol=10e-4)

    def test_instance_norm3d_6(self, dtype=torch.float):
        rand_input = torch.randn((1, 64, 112, 112, 80), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(64)
        cpu_result = test_module(rand_input)
        xpu_module = test_module.to("xpu")
        xpu_result = xpu_module(rand_input.to("xpu"))
        self.assertEqual(cpu_result, xpu_result.to("cpu"))

    def test_instance_norm3d_bfloat16_6(self, dtype=torch.bfloat16):
        rand_input = torch.randn((1, 64, 112, 112, 80), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(64)
        cpu_result = test_module(rand_input)
        xpu_module = test_module.to("xpu")
        xpu_result = xpu_module(rand_input.to("xpu"))
        self.assertEqual(cpu_result, xpu_result.to("cpu"), rtol=10e-3, atol=10e-4)

    def test_instance_norm3d_float16_6(self, dtype=torch.bfloat16):
        rand_input = torch.randn((1, 64, 112, 112, 80), dtype=dtype)
        test_module = torch.nn.InstanceNorm3d(64)
        cpu_result = test_module(rand_input)
        dtype_dpcpp = torch.float16
        xpu_module = test_module.to("xpu").to(dtype_dpcpp)
        xpu_result = xpu_module(rand_input.to("xpu").to(dtype_dpcpp))
        self.assertEqual(cpu_result, xpu_result.to("cpu").to(torch.bfloat16), rtol=10e-3, atol=10e-4)
