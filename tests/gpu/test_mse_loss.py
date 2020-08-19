import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestNNMethod(TestCase):
    @pytest.mark.skipif("torch_ipex._double_kernel_disabled()")    
    def test_mse_loss(self, dtype=torch.float):

        print('none')
        loss = nn.MSELoss(reduction="none")
        input = torch.randn(3, 5)
        input_dpcpp = input.to("dpcpp")
        target = torch.randn(3, 5)

        print("cpu")
        input_cpu = input
        target_cpu = target
        input_cpu.requires_grad = True
        output_cpu = loss(input_cpu, target_cpu)
        print(output_cpu)
        output_cpu.backward(torch.ones_like(target_cpu, dtype=torch.float))
        print(input_cpu.grad)

        print("dpcpp")

        target_dpcpp = target.to("dpcpp")
        input_dpcpp.requires_grad = True
        output_dpcpp = loss(input_dpcpp, target_dpcpp)
        print(output_dpcpp.cpu())
        output_dpcpp.backward(torch.ones_like(
            target, dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.cpu())
        self.assertEqual(output_dpcpp, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())
        input_cpu.grad.zero_()
        input_dpcpp.grad.zero_()

        print("sum")
        loss = nn.MSELoss(reduction="sum")

        print("cpu")
        input_cpu = input
        target_cpu = target
        output_cpu = loss(input_cpu, target_cpu)
        print(output_cpu)
        output_cpu.backward(torch.tensor((2.0), dtype=torch.float))
        print(input_cpu.grad)
        # input_cpu.grad.zero_()

        print("dpcpp")
        target_dpcpp = target.to("dpcpp")
        input_dpcpp.requires_grad = True
        output_dpcpp = loss(input_dpcpp, target_dpcpp)
        print(output_dpcpp.cpu())
        output_dpcpp.backward(torch.tensor(
            (2.0), dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.cpu())
        # input_dpcpp.grad.zero_()
        self.assertEqual(output_dpcpp, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())
        input_cpu.grad.zero_()
        input_dpcpp.grad.zero_()

        print("mean")
        loss = nn.MSELoss(reduction="mean")

        print("cpu")
        input_cpu = input
        target_cpu = target
        output_cpu = loss(input_cpu, target_cpu)
        print(output_cpu)
        output_cpu.backward(torch.tensor((2.0), dtype=torch.float))
        print(input_cpu.grad)
        # input_cpu.grad.zero_()

        print("dpcpp")
        target_dpcpp = target.to("dpcpp")
        input_dpcpp.requires_grad = True
        output_dpcpp = loss(input_dpcpp, target_dpcpp)
        print(output_dpcpp.cpu())
        output_dpcpp.backward(torch.tensor(
            (2.0), dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.cpu())
        # input_dpcpp.grad.zero_()
        self.assertEqual(output_dpcpp, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())
        input_cpu.grad.zero_()
        input_dpcpp.grad.zero_()
