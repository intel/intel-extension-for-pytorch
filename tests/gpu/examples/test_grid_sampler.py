import torch
import intel_extension_for_pytorch
import random
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)
from torch.nn import functional as F
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    @repeat_test_for_types([torch.float, torch.half, torch.bfloat16])
    def test_gridSampler(self, dtype=torch.float):
        inp = torch.ones(1, 1, 4, 4)
        out_h = 20
        out_w = 20

        new_h = torch.linspace(-1, 1, out_h).view(-1, 1).repeat(1, out_w)
        new_w = torch.linspace(-1, 1, out_w).repeat(out_h, 1)
        grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0)
        outp = F.grid_sample(inp, grid=grid, mode='bilinear', align_corners=True)
        print("cpu result ", outp)
        grid_xpu = grid.to("xpu")
        inp_xpu = inp.to("xpu")
        outp_xpu = F.grid_sample(inp_xpu, grid=grid_xpu, mode='bilinear', align_corners=True)
        print("xpu result ", outp_xpu.to("cpu"))
        self.assertEqual(outp.to(cpu_device), outp_xpu.to(cpu_device))

    @repeat_test_for_types([torch.float, torch.half, torch.bfloat16])
    def test_gridSampler_3d(self, dtype=torch.float):
        N = random.randint(2, 5)
        C = random.randint(2, 4)
        ID = random.randint(2, 5)
        IH = random.randint(2, 5)
        IW = random.randint(2, 5)
        D = random.randint(ID + 1, 7)
        H = random.randint(IH + 1, 7)
        W = random.randint(IW + 1, 7)
        input_cpu = torch.randn(C, N, ID, IH, IW).transpose(0, 1).requires_grad_()
        grid_cpu = torch.randn(D, N, H, W, 3).transpose(0, 1).requires_grad_()
        out_cpu = F.grid_sample(input_cpu, grid_cpu, mode='bilinear', align_corners=True)

        grid_xpu = grid_cpu.to(dpcpp_device)
        inp_xpu = input_cpu.to(dpcpp_device)
        out_xpu = F.grid_sample(inp_xpu, grid=grid_xpu, mode='bilinear', align_corners=True)
        self.assertEqual(out_cpu.to(cpu_device), out_xpu.to(cpu_device))

    @repeat_test_for_types([torch.float, torch.half, torch.bfloat16])
    def test_gridSampler_2d_Bicubic(self, dtype=torch.float):
        N = random.randint(2, 8)
        C = random.randint(2, 6)
        H = random.randint(2, 8)
        W = random.randint(2, 8)
        input_cpu = torch.randn(N, C, H, W, requires_grad=True)
        grid_cpu = torch.randn(N, H, W, 2, requires_grad=True)
        out_cpu = F.grid_sample(input_cpu, grid_cpu, mode='bicubic', align_corners=True)

        grid_xpu = grid_cpu.to(dpcpp_device)
        inp_xpu = input_cpu.to(dpcpp_device)
        out_xpu = F.grid_sample(inp_xpu, grid=grid_xpu, mode='bicubic', align_corners=True)
        print("out_cpu", out_cpu)
        print("out_xpu", out_xpu.cpu())
        self.assertEqual(out_cpu.to(cpu_device), out_xpu.to(cpu_device))
