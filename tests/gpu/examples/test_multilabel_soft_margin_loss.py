import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestNNMethod(TestCase):
    def test_multiabel_soft_margin_loss(self, dtype=torch.float):

        input = torch.randn(3, 5)
        target = torch.LongTensor(3, 5).random_(-1, 5)

        input_cpu = input
        target_cpu = target

        input_dpcpp = input.to("dpcpp")
        target_dpcpp = target.to("dpcpp")

        def _test_cpu(input, target, reduc):
            loss = nn.MultiLabelSoftMarginLoss(reduction=reduc)
            input.requires_grad = True
            output = loss(input, target)
            print(output)
            output.backward(torch.ones_like(output, dtype=torch.float))
            print(input.grad)
            try:
                return output, input
            finally:
                input.grad.zero_()

        def _test_dpcpp(input, target, reduc):
            loss = nn.MultiLabelSoftMarginLoss(reduction=reduc)
            input.requires_grad = True
            output = loss(input, target)
            print(output.cpu())
            output.backward(torch.ones_like(
                output, dtype=torch.float).to("dpcpp"))
            print(input.grad.cpu())
            try:
                return output, input
            finally:
                input.grad.zero_()

        print('none')
        print("cpu")
        output_cpu, input_cpu = _test_cpu(input_cpu, target_cpu, "none")
        print("dpcpp")
        output_dpcpp, input_dpcpp = _test_dpcpp(
            input_dpcpp, target_dpcpp, "none")
        self.assertEqual(output_cpu, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())

        print('sum')
        print("cpu")
        output_cpu, input_cpu = _test_cpu(input_cpu, target_cpu, "sum")
        print("dpcpp")
        output_dpcpp, input_dpcpp = _test_dpcpp(
            input_dpcpp, target_dpcpp, "sum")
        self.assertEqual(output_cpu, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())

        print('mean')
        print("cpu")
        output_cpu, input_cpu = _test_cpu(input_cpu, target_cpu, "mean")
        print("dpcpp")
        output_dpcpp, input_dpcpp = _test_dpcpp(
            input_dpcpp, target_dpcpp, "mean")
        self.assertEqual(output_cpu, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())

        print('sum-1024*1024')
        input = torch.randn(1024, 1024)
        target = torch.LongTensor(1024, 1024).random_(-1, 1024)

        input_cpu = input
        target_cpu = target

        input_dpcpp = input.to("dpcpp")
        target_dpcpp = target.to("dpcpp")

        print('cpu')
        output_cpu, input_cpu = _test_cpu(input_cpu, target_cpu, "sum")
        print('dpcpp')
        output_dpcpp, input_dpcpp = _test_dpcpp(
            input_dpcpp, target_dpcpp, "sum")
        tol = output_cpu.item()*5e-5
        self.assertEqual(output_cpu, output_dpcpp.cpu(),tol)
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())
