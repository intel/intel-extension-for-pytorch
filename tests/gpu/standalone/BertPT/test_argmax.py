import torch
from torch.testing._internal.common_utils import TestCase

import numpy as np

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_argmax_1(self, dtype=torch.float):
        t = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        t_cpu = torch.argmax(t)
        dst_t = t.clone().to(dpcpp_device)
        t_xpu = torch.argmax(dst_t)
        self.assertEqual(t_cpu, t_xpu.to(cpu_device))

    def test_argmax_bfloat16_1(self, dtype=torch.bfloat16):
        t = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        t_cpu = torch.argmax(t)
        dst_t = t.clone().to(dpcpp_device)
        t_xpu = torch.argmax(dst_t)
        self.assertEqual(t_cpu, t_xpu.to(cpu_device))

    def test_argmax_float16_1(self, dtype=torch.float16):
        t = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        t_cpu = torch.argmax(t)
        dst_t = t.clone().to(dpcpp_device)
        t_xpu = torch.argmax(dst_t)
        self.assertEqual(t_cpu, t_xpu.to(cpu_device))
    
    def test_argmax_dim_1(self, dtype=torch.float):
        t = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        t_cpu = torch.argmax(t, dim=1)
        dst_t = t.clone().to(dpcpp_device)
        t_xpu = torch.argmax(dst_t, dim=1)
        self.assertEqual(t_cpu, t_xpu.to(cpu_device))

    def test_argmax_dim_bfloat16_1(self, dtype=torch.bfloat16):
        t = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        t_cpu = torch.argmax(t, dim=1)
        dst_t = t.clone().to(dpcpp_device)
        t_xpu = torch.argmax(dst_t, dim=1)
        self.assertEqual(t_cpu, t_xpu.to(cpu_device))

    def test_argmax_dim_float16_1(self, dtype=torch.float16):
        t = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        t_cpu = torch.argmax(t, dim=1)
        dst_t = t.clone().to(dpcpp_device)
        t_xpu = torch.argmax(dst_t, dim=1)
        self.assertEqual(t_cpu, t_xpu.to(cpu_device))

    def test_argmax_2(self, dtype=torch.float):
        t = torch.randn((20, 30522), device=cpu_device, dtype=dtype)
        t_cpu = torch.argmax(t)
        dst_t = t.clone().to(dpcpp_device)
        t_xpu = torch.argmax(dst_t)
        self.assertEqual(t_cpu, t_xpu.to(cpu_device))

    def test_argmax_bfloat16_2(self, dtype=torch.bfloat16):
        t = torch.randn((20, 30522), device=cpu_device, dtype=dtype)
        t_cpu = torch.argmax(t)
        dst_t = t.clone().to(dpcpp_device)
        t_xpu = torch.argmax(dst_t)
        self.assertEqual(t_cpu, t_xpu.to(cpu_device))

    def test_argmax_float16_2(self, dtype=torch.float16):
        t = torch.randn((20, 30522), device=cpu_device, dtype=dtype)
        t_cpu = torch.argmax(t)
        dst_t = t.clone().to(dpcpp_device)
        t_xpu = torch.argmax(dst_t)
        self.assertEqual(t_cpu, t_xpu.to(cpu_device))
    
    def test_argmax_dim_2(self, dtype=torch.float):
        t = torch.randn((20, 30522), device=cpu_device, dtype=dtype)
        t_cpu = torch.argmax(t, dim=1)
        dst_t = t.clone().to(dpcpp_device)
        t_xpu = torch.argmax(dst_t, dim=1)
        self.assertEqual(t_cpu, t_xpu.to(cpu_device))

    def test_argmax_dim_bfloat16_2(self, dtype=torch.bfloat16):
        t = torch.randn((20, 30522), device=cpu_device, dtype=dtype)
        t_cpu = torch.argmax(t, dim=1)
        dst_t = t.clone().to(dpcpp_device)
        t_xpu = torch.argmax(dst_t, dim=1)
        self.assertEqual(t_cpu, t_xpu.to(cpu_device))

    def test_argmax_dim_float16_2(self, dtype=torch.float16):
        t = torch.randn((20, 30522), device=cpu_device, dtype=dtype)
        t_cpu = torch.argmax(t, dim=1)
        dst_t = t.clone().to(dpcpp_device)
        t_xpu = torch.argmax(dst_t, dim=1)
        self.assertEqual(t_cpu, t_xpu.to(cpu_device))
        
