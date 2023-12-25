import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

shapes = [
    ((1, 2048), (1000, 2048)),
    ((2, 384, 1024), (2, 1024)),
    ((2, 384, 1024), (1024, 1024)),
    ((2, 384, 1024), (4096, 1024)),
    ((2, 384, 4096), (1024, 4096)),
]

class TestNNMethod(TestCase):
    def test_linear(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape[0], "==================")
            # cpu
            linear = nn.Linear(shape[1][1], shape[1][0], bias=True)
            x_cpu = torch.randn(
                (shape[0]),
                requires_grad=True,
                dtype=dtype,
            )
    
            z_cpu = linear(x_cpu)
            linear.zero_grad()

            # dpcpp
            linear_dpcpp = linear.to("xpu")
            x_dpcpp = x_cpu.to("xpu")
            z_dpcpp = linear_dpcpp(x_dpcpp)
            self.assertEqual(z_cpu, z_dpcpp)

    def test_linear_bfloat16(self, dtype=torch.bfloat16):
        for shape in shapes:
            print("\n================== test shape: ", shape[0], "==================")
            # cpu
            linear = nn.Linear(shape[1][1], shape[1][0], bias=True, dtype=dtype)
            x_cpu = torch.randn(
                (shape[0]),
                requires_grad=True,
                dtype=dtype,
            )
            
            z_cpu = linear(x_cpu)
            linear.zero_grad()

            # dpcpp
            linear_dpcpp = linear.to("xpu")
            x_dpcpp = x_cpu.to("xpu")
            z_dpcpp = linear_dpcpp(x_dpcpp)
            self.assertEqual(z_cpu, z_dpcpp)

    def test_linear_float16(self, dtype=torch.bfloat16):
        for shape in shapes:
            print("\n================== test shape: ", shape[0], "==================")
            # cpu
            linear = nn.Linear(shape[1][1], shape[1][0], bias=True, dtype=dtype)
            x_cpu = torch.randn(
                (shape[0]),
                requires_grad=True,
                dtype=dtype,
            )
            
            z_cpu = linear(x_cpu)
            linear.zero_grad()
            dtype_dpcpp = torch.float16
            # dpcpp
            linear_dpcpp = linear.to("xpu").to(dtype_dpcpp)
            x_dpcpp = x_cpu.to("xpu").to(dtype_dpcpp)
            z_dpcpp = linear_dpcpp(x_dpcpp)
            self.assertEqual(z_cpu, z_dpcpp.cpu().to(torch.bfloat16))
           
