import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestNNMethod(TestCase):
    def test_soft_margin_loss(self, dtype=torch.float):
        input = torch.randn(3, 5)
        target = torch.randn(3, 5)

        input_cpu = input
        target_cpu = target

        input_dpcpp = input.to("dpcpp")
        target_dpcpp = target.to("dpcpp")

        def _test_cpu(input, target, reduc):
            loss = nn.SoftMarginLoss(reduction=reduc)
            input.requires_grad = True
            output = loss(input, target)
            print(output)
            if(reduc == "none"):
                output.backward(torch.ones_like(input, dtype=torch.float))
            else:
                output.backward(torch.tensor((1.0), dtype=torch.float))
            print(input.grad)
            try:
                return input, output
            finally:
                input.grad.zero_()

        def _test_dpcpp(input, target, reduc):
            loss = nn.SoftMarginLoss(reduction=reduc)
            input.requires_grad = True
            output = loss(input, target)
            print(output.cpu())
            if(reduc == "none"):
                output.backward(torch.ones_like(
                    input, dtype=torch.float).to("dpcpp"))
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
        input_cpu, output_cpu = _test_cpu(input_cpu, target_cpu, "none")
        print("dpcpp")
        input_dpcpp, out_dpcpp = _test_dpcpp(input_dpcpp, target_dpcpp, "none")
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())
        self.assertEqual(output_cpu, out_dpcpp.cpu())

        print('sum')
        print("cpu")
        input_cpu, output_cpu = _test_cpu(input_cpu, target_cpu, "sum")
        print("dpcpp")
        input_dpcpp, out_dpcpp = _test_dpcpp(input_dpcpp, target_dpcpp, "sum")
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())
        self.assertEqual(output_cpu, out_dpcpp.cpu())

        print('mean')
        print("cpu")
        input_cpu, output_cpu = _test_cpu(input_cpu, target_cpu, "mean")
        print("dpcpp")
        input_dpcpp, out_dpcpp = _test_dpcpp(input_dpcpp, target_dpcpp, "mean")
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())
        self.assertEqual(output_cpu, out_dpcpp.cpu())
