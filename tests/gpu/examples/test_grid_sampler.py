import torch
import ipex
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)
from torch.nn import functional as F

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
