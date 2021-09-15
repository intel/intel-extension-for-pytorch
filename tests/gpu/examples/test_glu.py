import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_glu(self, dtype=torch.float):
        input_cpu = torch.randn(4, 6)
        input_dpcpp = input_cpu.to("xpu")
        m = nn.GLU()

        print("cpu")
        input_cpu.requires_grad = True
        output_cpu = m(input_cpu)
        print("output: ", output_cpu)
        output_cpu.backward(torch.ones_like(output_cpu))
        print("input.grad: ", input_cpu.grad)

        print("xpu")
        input_dpcpp.requires_grad = True
        output_dpcpp = m(input_dpcpp)
        print("output: ", output_dpcpp.cpu())
        output_dpcpp.backward(torch.ones_like(output_dpcpp).to("xpu"))
        print("input.grad: ", input_dpcpp.grad.cpu())
        self.assertEqual(output_cpu, output_dpcpp)
        self.assertEqual(input_cpu.grad, input_dpcpp.grad)
