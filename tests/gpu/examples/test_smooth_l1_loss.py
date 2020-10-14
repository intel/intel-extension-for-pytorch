import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    def test_smooth_l1_loss(self, dtype=torch.float):

        print('none')
        loss = nn.SmoothL1Loss(reduction="none")
        input = torch.randn(3, 5, requires_grad=True)
        target = torch.randn(3, 5)

        print("cpu")
        input_cpu = input
        target_cpu = target
        output_cpu = loss(input_cpu, target_cpu)
        print(output_cpu)
        output_cpu.backward(torch.ones_like(target_cpu, dtype=torch.float))
        print(input_cpu.grad)

        print("dpcpp")
        input_dpcpp = input
        target_dpcpp = target
        output_dpcpp = loss(input_dpcpp.to("dpcpp"), target_dpcpp.to("dpcpp"))
        print(output_dpcpp.cpu())
        output_dpcpp.backward(torch.ones_like(
            target_dpcpp, dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.cpu())
        self.assertEqual(input_cpu, input_cpu.cpu())
        self.assertEqual(output_cpu, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())
        input_cpu.grad.zero_()
        input_dpcpp.grad.zero_()

        print("sum")
        loss = nn.SmoothL1Loss(reduction="sum")

        print("cpu")
        input_cpu = input
        target_cpu = target
        output_cpu = loss(input_cpu, target_cpu)
        print(output_cpu)
        output_cpu.backward(torch.tensor((2.0), dtype=torch.float))
        print(input_cpu.grad)
        # input_cpu.grad.zero_()

        print("dpcpp")
        input_dpcpp = input
        target_dpcpp = target
        output_dpcpp = loss(input_dpcpp.to("dpcpp"), target_dpcpp.to("dpcpp"))
        print(output_dpcpp.cpu())
        output_dpcpp.backward(torch.tensor(
            (2.0), dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.cpu())
        # input_dpcpp.grad.zero_()
        self.assertEqual(input_cpu, input_cpu.cpu())
        self.assertEqual(output_cpu, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())
        input_cpu.grad.zero_()
        input_dpcpp.grad.zero_()

        print("mean")
        loss = nn.SmoothL1Loss(reduction="mean")

        print("cpu")
        input_cpu = input
        target_cpu = target
        output_cpu = loss(input_cpu, target_cpu)
        print(output_cpu)
        output_cpu.backward(torch.tensor((2.0), dtype=torch.float))
        print(input_cpu.grad)
        # input_cpu.grad.zero_()

        print("dpcpp")
        input_dpcpp = input
        target_dpcpp = target
        output_dpcpp = loss(input_dpcpp.to("dpcpp"), target_dpcpp.to("dpcpp"))
        print(output_dpcpp.cpu())
        output_dpcpp.backward(torch.tensor(
            (2.0), dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.cpu())
        # input_dpcpp.grad.zero_()
        self.assertEqual(input_cpu, input_cpu.cpu())
        self.assertEqual(output_cpu, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())
        input_cpu.grad.zero_()
        input_dpcpp.grad.zero_()
