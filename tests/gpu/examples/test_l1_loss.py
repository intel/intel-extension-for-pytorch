import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_l1_loss(self, dtype=torch.float):

        print('none')
        loss = nn.L1Loss(reduction="none")
        input = torch.randn(3, 5, requires_grad=True)
        target = torch.randn(3, 5)

        print("cpu")
        input_cpu = input
        target_cpu = target
        output_cpu = loss(input_cpu, target_cpu)
        print(output_cpu)
        output_cpu.backward(torch.ones_like(target_cpu, dtype=torch.float))
        print(input_cpu.grad)
        input_cpu.grad.zero_()

        print("xpu")
        input_dpcpp = input
        target_dpcpp = target
        output_dpcpp = loss(input_dpcpp.to("xpu"), target_dpcpp.to("xpu"))
        print(output_dpcpp.cpu())
        output_dpcpp.backward(torch.ones_like(
            target_dpcpp, dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.cpu())
        input_dpcpp.grad.zero_()
        self.assertEqual(input, input_dpcpp.to(cpu_device))
        self.assertEqual(output_cpu, output_dpcpp.to(cpu_device))

        print('sum')
        loss = nn.L1Loss(reduction="sum")

        print("cpu")
        input_cpu = input
        target_cpu = target
        output_cpu = loss(input_cpu, target_cpu)
        print(output_cpu)
        output_cpu.backward(torch.tensor((2.0), dtype=torch.float))
        print(input_cpu.grad)
        input_cpu.grad.zero_()

        print("xpu")
        input_dpcpp = input
        target_dpcpp = target
        output_dpcpp = loss(input_dpcpp.to("xpu"), target_dpcpp.to("xpu"))
        print(output_dpcpp.cpu())
        output_dpcpp.backward(torch.tensor(
            (2.0), dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.cpu())
        input_dpcpp.grad.zero_()
        self.assertEqual(input, input_dpcpp.to(cpu_device))
        self.assertEqual(output_cpu, output_dpcpp.to(cpu_device))

        print('mean')
        loss = nn.L1Loss(reduction="mean")

        print("cpu")
        input_cpu = input
        target_cpu = target
        output_cpu = loss(input_cpu, target_cpu)
        print(output_cpu)
        output_cpu.backward(torch.tensor((2.0), dtype=torch.float))
        print(input_cpu.grad)
        input_cpu.grad.zero_()

        print("xpu")
        input_dpcpp = input
        target_dpcpp = target
        output_dpcpp = loss(input_dpcpp.to("xpu"), target_dpcpp.to("xpu"))
        print(output_dpcpp.cpu())
        output_dpcpp.backward(torch.tensor(
            (2.0), dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.cpu())
        input_dpcpp.grad.zero_()
        self.assertEqual(input, input_dpcpp.to(cpu_device))
        self.assertEqual(output_cpu, output_dpcpp.to(cpu_device))
