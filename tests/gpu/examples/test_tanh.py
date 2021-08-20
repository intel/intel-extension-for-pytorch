import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import ipex


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_tanh(self, dtype=torch.float):

        # cpu
        tanh = nn.Tanh()

        x_cpu = torch.tensor([[1.23, 2.34, 6.45, 2.22], [
            0.23, 1.34, 7.45, 1.22]], requires_grad=True, device=cpu_device, dtype=dtype)
        print("x_cpu", x_cpu)

        z_cpu = tanh(x_cpu)
        print("z_cpu", z_cpu)

        z_cpu.backward(torch.tensor(
            [[1, 1, 1, 1], [2, 2, 3, 4]], device=cpu_device, dtype=dtype))
        print("cpu input grad", x_cpu.grad)

        # dpcpp
        x_dpcpp = torch.tensor([[1.23, 2.34, 6.45, 2.22], [
            0.23, 1.34, 7.45, 1.22]], requires_grad=True, device=dpcpp_device, dtype=dtype)
        print("x_dpcpp", x_dpcpp.to("cpu"))

        z_dpcpp = tanh(x_dpcpp)
        print("z_dpcpp", z_dpcpp.to("cpu"))

        z_dpcpp.backward(torch.tensor(
            [[1, 1, 1, 1], [2, 2, 3, 4]], device=dpcpp_device, dtype=dtype))
        print("dpcpp input grad", x_dpcpp.grad.to("cpu"))
        self.assertEqual(x_cpu, x_dpcpp.cpu())
        self.assertEqual(z_cpu, z_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
