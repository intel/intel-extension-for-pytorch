import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch
import pytest

cpu_device = torch.device("cpu")

def double_step_seq(step1, len1, step2, len2):
    seq1 = torch.arange(0, step1 * len1, step1)
    seq2 = torch.arange(0, step2 * len2, step2)
    return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

class TestTorchMethod(TestCase):
    def test_index_fill(self, dtype=torch.float):
        x = torch.randn((8192, 8192), device=cpu_device)
        index = torch.linspace(0, 8190, steps=4096, device=cpu_device).to(torch.long)
        y = x.index_fill(0, index, 0)
        print("y = ", y)

        x_xpu = x.xpu()
        index = index.xpu()
        y_xpu = x_xpu.index_fill(0, index, 0)
        print("y_xpu = ", y_xpu.cpu())
        self.assertEqual(y, y_xpu.cpu())

    def test_index_copy(self, dtype=torch.float):
        x = torch.randn((8192, 8192), device=cpu_device)
        t = torch.randn((4096, 8192), dtype=torch.float)
        index = torch.linspace(0, 8190, steps=4096, device=cpu_device).to(torch.long)
        y = x.index_copy(0, index, t)
        print("y = ", y)

        x_xpu = x.xpu()
        t = t.xpu()
        index = index.xpu()
        y_xpu = x_xpu.index_copy(0, index, t)
        print("y_xpu = ", y_xpu.cpu())
        self.assertEqual(y, y_xpu.cpu())

    def test_index_add(self, dtype=torch.float):
        x = torch.randn((8192, 8192), device=cpu_device)
        t = torch.randn((4096, 8192), dtype=torch.float)
        index = torch.linspace(0, 8190, steps=4096, device=cpu_device).to(torch.long)
        y = x.index_add(0, index, t)
        print("y = ", y)

        x_xpu = x.xpu()
        t = t.xpu()
        index = index.xpu()
        y_xpu = x_xpu.index_add(0, index, t)
        print("y_xpu = ", y_xpu.cpu())
        self.assertEqual(y, y_xpu.cpu())

    def test_index_2_dim(self, dtype=torch.float):
        table = torch.randn([169, 16])
        table_xpu = table.to("xpu")
        rel_index_coords = double_step_seq(13, 7, 1, 7)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()

        out_cpu = table[rel_position_index.view(-1)]
        out_xpu = table_xpu[rel_position_index.view(-1)]
        self.assertEqual(out_cpu, out_xpu.cpu())
