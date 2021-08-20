import numpy
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestNNMethod(TestCase):
  def test_flip(self, dtype=torch.float):
    x_cpu = torch.arange(8, dtype=dtype, device=cpu_device).view(2, 2, 2)
    x_dpcpp = x_cpu.to(dpcpp_device)
    y_cpu = x_cpu.flip([0, 1])
    y_dpcpp = x_dpcpp.flip([0, 1])
    print("test_flip:")
    print("cpu:")
    print(y_cpu)
    print("dpcpp:")
    print(y_dpcpp.cpu())
    self.assertEqual(y_cpu, y_dpcpp.cpu())

  def test_roll(self, dtype=torch.float):
    x_cpu = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtype, device=cpu_device).view(4, 2)
    x_dpcpp = x_cpu.to(dpcpp_device)
    # test 1
    y_cpu = x_cpu.roll(1, 0)
    y_dpcpp = x_dpcpp.roll(1, 0)
    print("test_roll(case 1):")
    print("cpu:")
    print(y_cpu)
    print("dpcpp:")
    print(y_dpcpp.cpu())
    self.assertEqual(y_cpu, y_dpcpp.cpu())
    # test 2
    y_cpu = x_cpu.roll(-1, 0)
    y_dpcpp = x_dpcpp.roll(-1, 0)
    print("test_roll(case 2):")
    print("cpu:")
    print(y_cpu)
    print("dpcpp:")
    print(y_dpcpp.cpu())
    self.assertEqual(y_cpu, y_dpcpp.cpu())
    # test 3
    y_cpu = x_cpu.roll(shifts=(2, 1), dims=(0, 1))
    y_dpcpp = x_dpcpp.roll(shifts=(2, 1), dims=(0, 1))
    print("test_roll(case 3):")
    print("cpu:")
    print(y_cpu)
    print("dpcpp:")
    print(y_dpcpp.cpu())
    self.assertEqual(y_cpu, y_dpcpp.cpu())

  def test_rot90(self, dtype=torch.float):
    x_cpu = torch.arange(8, dtype=dtype, device=cpu_device).view(2, 2, 2)
    x_dpcpp = x_cpu.to(dpcpp_device)
    y_cpu = x_cpu.rot90(1, [1, 2])
    y_dpcpp = x_dpcpp.rot90(1, [1, 2])
    print("test_rot90:")
    print("cpu:")
    print(y_cpu)
    print("dpcpp:")
    print(y_dpcpp.cpu())
    self.assertEqual(y_cpu, y_dpcpp.cpu())
