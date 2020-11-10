import torch
from torch.nn.functional import relu_
from torch.nn.functional import relu
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_relu(self, dtype=torch.float):

        x_cpu = torch.tensor(
            [[-0.1, 0.2], [-0.2, 0.3], [0.4, 0.5], [0.5, -0.6]])
        x_dpcpp = x_cpu.to("xpu")

        relu_(x_cpu)
        relu_(x_dpcpp)
        print("cpu relu_ ", x_cpu)
        print("dpcpp relu_ ", x_dpcpp.cpu())
        self.assertEqual(x_cpu, x_dpcpp.cpu())

        x_cpu.requires_grad_(True)
        x_dpcpp.requires_grad_(True)
        y_cpu = relu(x_cpu)
        y_dpcpp = relu(x_dpcpp)
        print("cpu relu ", y_cpu)
        print("dpcpp relu ", y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())

        y_cpu.backward(x_cpu)
        y_dpcpp.backward(y_dpcpp)

        print("cpu relu bwd", x_cpu.grad)
        print("dpcpp relu bwd", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
