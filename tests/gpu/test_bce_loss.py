import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestNNMethod(TestCase):
    def test_bce_loss(self, dtype=torch.float):

        print('none')
        loss = nn.BCELoss(reduction="none")
        input = torch.tensor([0.2, 0.7, 0.9], requires_grad=True)
        target = torch.tensor([0.5, 0.5, 0.5])

        print("cpu")
        output_cpu = loss(input, target)
        print(output_cpu)
        output_cpu.backward(torch.ones_like(target, dtype=torch.float))
        print(input.grad)

        print("dpcpp")
        loss.to("dpcpp")
        input_dpcpp = torch.tensor(
            [0.2, 0.7, 0.9], device=dpcpp_device, requires_grad=True)
        output_dpcpp = loss(input_dpcpp, target.to("dpcpp"))
        print(output_dpcpp.to("cpu"))
        output_dpcpp.backward(torch.ones_like(
            target, dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.to("cpu"))
        self.assertEqual(input.grad, input_dpcpp.grad.to(cpu_device))
        self.assertEqual(output_cpu, output_dpcpp.to(cpu_device))

        print('none (weight)')
        w = torch.tensor([0.2, 0.2, 0.2])
        loss = nn.BCELoss(weight=w, reduction="none")
        input = torch.tensor([0.2, 0.7, 0.9], requires_grad=True)
        target = torch.tensor([0.5, 0.5, 0.5])

        print("cpu")
        output_cpu = loss(input, target)
        print(output_cpu)
        output_cpu.backward(torch.ones_like(target, dtype=torch.float))
        print(input.grad)

        print("dpcpp")
        loss.to("dpcpp")
        input_dpcpp = torch.tensor(
            [0.2, 0.7, 0.9], device=dpcpp_device, requires_grad=True)
        output_dpcpp = loss(input_dpcpp, target.to("dpcpp"))
        print(output_dpcpp.to("cpu"))
        output_dpcpp.backward(torch.ones_like(
            target, dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.to("cpu"))
        self.assertEqual(input.grad, input_dpcpp.grad.to(cpu_device))
        self.assertEqual(output_cpu, output_dpcpp.to(cpu_device))

        print('sum')
        loss = nn.BCELoss(reduction="sum")
        input = torch.tensor([0.2, 0.7, 0.9], requires_grad=True)
        target = torch.tensor([0.5, 0.5, 0.5])

        print("cpu")
        output_cpu = loss(input, target)
        print(output_cpu)
        output_cpu.backward(torch.tensor((2.0), dtype=torch.float))
        print(input.grad)

        print("dpcpp")
        loss.to("dpcpp")
        input_dpcpp = torch.tensor(
            [0.2, 0.7, 0.9], device=dpcpp_device, requires_grad=True)
        output_dpcpp = loss(input_dpcpp, target.to("dpcpp"))
        print(output_dpcpp.to("cpu"))
        output_dpcpp.backward(torch.tensor(
            (2.0), dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.to("cpu"))
        self.assertEqual(input.grad, input_dpcpp.grad.to(cpu_device))
        self.assertEqual(output_cpu, output_dpcpp.to(cpu_device))

        print('sum (weight)')
        w = torch.tensor([0.2, 0.2, 0.2])
        loss = nn.BCELoss(weight=w, reduction="sum")
        input = torch.tensor([0.2, 0.7, 0.9], requires_grad=True)
        target = torch.tensor([0.5, 0.5, 0.5])

        print("cpu")
        output_cpu = loss(input, target)
        print(output_cpu)
        output_cpu.backward(torch.tensor((0.5), dtype=torch.float))
        print(input.grad)

        print("dpcpp")
        loss.to("dpcpp")
        input_dpcpp = torch.tensor(
            [0.2, 0.7, 0.9], device=dpcpp_device, requires_grad=True)
        output_dpcpp = loss(input_dpcpp, target.to("dpcpp"))
        print(output_dpcpp.to("cpu"))
        output_dpcpp.backward(torch.tensor(
            (0.5), dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.to("cpu"))
        self.assertEqual(input.grad, input_dpcpp.grad.to(cpu_device))
        self.assertEqual(output_cpu, output_dpcpp.to(cpu_device))

        print('mean')
        loss = nn.BCELoss(reduction="mean")
        input = torch.tensor([0.2, 0.7, 0.9], requires_grad=True)
        target = torch.tensor([0.5, 0.5, 0.5])

        print("cpu")
        output_cpu = loss(input, target)
        print(output_cpu)
        output_cpu.backward(torch.tensor((2.0), dtype=torch.float))
        print(input.grad)

        print("dpcpp")
        loss.to("dpcpp")
        input_dpcpp = torch.tensor(
            [0.2, 0.7, 0.9], device=dpcpp_device, requires_grad=True)
        output_dpcpp = loss(input_dpcpp, target.to("dpcpp"))
        print(output_dpcpp.to("cpu"))
        output_dpcpp.backward(torch.tensor(
            (2.0), dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.to("cpu"))
        self.assertEqual(input.grad, input_dpcpp.grad.to(cpu_device))
        self.assertEqual(output_cpu, output_dpcpp.to(cpu_device))

        print('mean (weight)')
        w = torch.tensor([0.2, 0.2, 0.2])
        loss = nn.BCELoss(weight=w, reduction="mean")
        input = torch.tensor([0.2, 0.7, 0.9], requires_grad=True)
        target = torch.tensor([0.5, 0.5, 0.5])

        print("cpu")
        output_cpu = loss(input, target)
        print(output_cpu)
        output_cpu.backward(torch.tensor((0.5), dtype=torch.float))
        print(input.grad)

        print("dpcpp")
        loss.to("dpcpp")
        input_dpcpp = torch.tensor(
            [0.2, 0.7, 0.9], device=dpcpp_device, requires_grad=True)
        output_dpcpp = loss(input_dpcpp, target.to("dpcpp"))
        print(output_dpcpp.to("cpu"))
        output_dpcpp.backward(torch.tensor(
            (0.5), dtype=torch.float, device=dpcpp_device))
        print(input_dpcpp.grad.to("cpu"))
        self.assertEqual(input.grad, input_dpcpp.grad.to(cpu_device))
        self.assertEqual(output_cpu, output_dpcpp.to(cpu_device))
