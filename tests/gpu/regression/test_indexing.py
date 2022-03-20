import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch
import pytest

cpu_device = torch.device("cpu")

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
