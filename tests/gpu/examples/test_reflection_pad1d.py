import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import ipex


class TestNNMethod(TestCase):
    def test_reflection_pad1d(self, dtype=torch.float):
        x_cpu = torch.arange(8, dtype=dtype).reshape(1, 2, 4)
        grad_cpu = torch.arange(16, dtype=dtype).reshape(1, 2, 8)
        x_dpcpp = x_cpu.to("xpu")
        grad_dpcpp = grad_cpu.to("xpu")
        m = nn.ReflectionPad1d(2)

        x_cpu.requires_grad_(True)
        y_cpu = m(x_cpu)
        print("y_cpu", y_cpu)
        output_cpu = y_cpu.backward(grad_cpu)
        print("x_cpu.grad", x_cpu.grad)

        m.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = m(x_dpcpp)
        print("y_dpcpp", y_dpcpp.to("cpu"))
        output_dpcpp = y_dpcpp.backward(grad_dpcpp)
        print("x_dpcpp.grad", x_dpcpp.grad.to("cpu"))

        self.assertEqual(y_cpu, y_dpcpp.to("cpu"))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to("cpu"))
