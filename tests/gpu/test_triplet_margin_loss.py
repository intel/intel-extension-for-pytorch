import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex
cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestNNMethod(TestCase):
    def test_triplet_margin_loss(self, dtype=torch.float):

        input = torch.randn(5, 6)
        positive = torch.randn(5, 6)
        negative = torch.randn(5, 6)

        input_cpu = input
        posit_cpu = positive
        negat_cpu = negative

        input_dpcpp = input.to("dpcpp")
        posit_dpcpp = positive.to("dpcpp")
        negat_dpcpp = negative.to("dpcpp")

        def _test_cpu(input, positive, negative, reduc):
            loss = nn.TripletMarginLoss(reduction=reduc)
            input.requires_grad = True
            output = loss(input, positive, negative)
            print(output)
            if(reduc == "none"):
                output.backward(torch.ones(5, dtype=torch.float))
            else:
                output.backward(torch.tensor((1.0), dtype=torch.float))
            print(input.grad)
            try:
                return input, output
            finally:
                input.grad.zero_()

        def _test_dpcpp(input, positive, negative, reduc):
            loss = nn.TripletMarginLoss(reduction=reduc)
            input.requires_grad = True
            output = loss(input, positive, negative)
            print(output.cpu())
            if(reduc == "none"):
                output.backward(torch.ones(5, dtype=torch.float).to("dpcpp"))
            else:
                output.backward(torch.tensor(
                    (1.0), dtype=torch.float).to("dpcpp"))
            print(input.grad.cpu())
            try:
                return input, output
            finally:
                input.grad.zero_()

        print('none')
        print("cpu")
        input_cpu, output_cpu = _test_cpu(
            input_cpu, posit_cpu, negat_cpu, "none")
        print("dpcpp")
        input_dpcpp, output_dpcpp = _test_dpcpp(
            input_dpcpp, posit_dpcpp, negat_dpcpp, "none")
        self.assertEqual(output_cpu, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())

        print('sum')
        print("cpu")
        input_cpu, output_cpu = _test_cpu(
            input_cpu, posit_cpu, negat_cpu, "sum")
        print("dpcpp")
        input_dpcpp, output_dpcpp = _test_dpcpp(
            input_dpcpp, posit_dpcpp, negat_dpcpp, "sum")
        self.assertEqual(output_cpu, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())

        print('mean')
        print("cpu")
        input_cpu, output_cpu = _test_cpu(
            input_cpu, posit_cpu, negat_cpu, "mean")
        print("dpcpp")
        input_dpcpp, output_dpcpp = _test_dpcpp(
            input_dpcpp, posit_dpcpp, negat_dpcpp, "mean")
        self.assertEqual(output_cpu, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())
