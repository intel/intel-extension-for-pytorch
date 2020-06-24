import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestNNMethod(TestCase):
    def test_glu(self, dtype=torch.float):
        input_cpu = torch.randn(4, 6)
        input_dpcpp = input_cpu.to("dpcpp")
        m = nn.GLU()

        print("cpu")
        input_cpu.requires_grad = True
        output_cpu = m(input_cpu)
        print("output: ", output_cpu)
        output_cpu.backward(torch.ones_like(output_cpu))
        print("input.grad: ", input_cpu.grad)
        input_cpu.grad.zero_()

        print("dpcpp")
        input_dpcpp.requires_grad = True
        output_dpcpp = m(input_dpcpp)
        print("output: ", output_dpcpp.cpu())
        output_dpcpp.backward(torch.ones_like(output_dpcpp).to("dpcpp"))
        print("input.grad: ", input_dpcpp.grad.cpu())
        input_dpcpp.grad.zero_()
