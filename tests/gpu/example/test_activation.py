import torch
import torch.nn.functional
import torch_ipex
from torch.testing._internal.common_utils import TestCase

class  TestNNMethod(TestCase):
    def test_activation(self, dtype=torch.float):

        relu_ = torch.nn.functional.relu_
        relu = torch.nn.functional.relu
        x_cpu = torch.tensor([[-0.1, 0.2],[-0.2, 0.3],[0.4, 0.5],[0.5, -0.6]]);
        x_dpcpp = x_cpu.to("dpcpp")

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
        y_dpcpp.backward(x_dpcpp)

        print("cpu relu bwd", x_cpu.grad)
        print("dpcpp relu bwd", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

        RReLU = torch.nn.RReLU(0.1,0.3)
        x_cpu = torch.tensor([[-0.1, 0.2],[-0.2, 0.3],[0.4, 0.5],[0.5, -0.6]]);
        x_dpcpp = x_cpu.to("dpcpp")
        x_cpu.requires_grad_(True)
        x_dpcpp.requires_grad_(True)
        y_cpu = RReLU(x_cpu)
        y_dpcpp = RReLU(x_dpcpp)
        print("cpu rrelu ", y_cpu)
        print("dpcpp rrelu ", y_dpcpp.cpu())
        #self.assertEqual(y_cpu, y_dpcpp.cpu())

        y_cpu.backward(x_cpu)
        y_dpcpp.backward(x_dpcpp)

        print("cpu rrelu bwd", x_cpu.grad)
        print("dpcpp rrelu bwd", x_dpcpp.grad.cpu())
        #self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

        GELU = torch.nn.GELU()
        x_cpu = torch.tensor([[-0.1, 0.2],[-0.2, 0.3],[0.4, 0.5],[0.5, -0.6]]);
        x_dpcpp = x_cpu.to("dpcpp")
        x_cpu.requires_grad_(True)
        x_dpcpp.requires_grad_(True)
        y_cpu = GELU(x_cpu)
        y_dpcpp = GELU(x_dpcpp)
        print("cpu gelu ", y_cpu)
        print("dpcpp gelu ", y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())

        #y_cpu = torch.tensor([[1, 1],[1, 1],[1, 1],[1, 1]]);
        #y_dpcpp = y_cpu.to("dpcpp")
        y_cpu.backward(x_cpu)
        y_dpcpp.backward(x_dpcpp)

        print("cpu gelu bwd", x_cpu.grad)
        print("dpcpp gelu bwd", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

        PReLU = torch.nn.PReLU()
        x_cpu = torch.tensor([[-0.1, 0.2],[-0.2, 0.3],[0.4, 0.5],[0.5, -0.6]]);
        x_dpcpp = x_cpu.to("dpcpp")
        x_cpu.requires_grad_(True)
        x_dpcpp.requires_grad_(True)
        y_cpu = PReLU(x_cpu)
        y_dpcpp = PReLU(x_dpcpp)
        print("cpu prelu ", y_cpu)
        print("dpcpp prelu ", y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())

        y_cpu.backward(x_cpu)
        y_dpcpp.backward(x_dpcpp)

        print("cpu prelu bwd", x_cpu.grad)
        print("dpcpp prelu bwd", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
