import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_multi_margin_loss(self, dtype=torch.float):

        input = torch.randn(3, 5)
        target = torch.LongTensor(3).random_(0, 5)

        input_cpu = input
        target_cpu = target

        input_dpcpp = input.to("xpu")
        target_dpcpp = target.to("xpu")

        def _test_cpu(input, target, reduc):
            loss = nn.MultiMarginLoss(reduction=reduc)
            input.requires_grad = True
            output = loss(input, target)
            print(output)
            if(reduc == "none"):
                output.backward(torch.ones_like(target, dtype=torch.float))
            else:
                output.backward(torch.tensor((1.0), dtype=torch.float))
            print(input.grad)
            try:
                return output, input
            finally:
                input.grad.zero_()

        def _test_dpcpp(input, target, reduc):
            loss = nn.MultiMarginLoss(reduction=reduc)
            input.requires_grad = True
            output = loss(input, target)
            print(output.cpu())
            if(reduc == "none"):
                output.backward(torch.ones_like(
                    target, dtype=torch.float).to("xpu"))
            else:
                output.backward(torch.tensor(
                    (1.0), dtype=torch.float).to("xpu"))
            print(input.grad.cpu())
            try:
                return output, input
            finally:
                input.grad.zero_()

        print('none')
        print("cpu")
        output_cpu, input_cpu = _test_cpu(input_cpu, target_cpu, "none")
        print("xpu")
        output_dpcpp, input_dpcpp = _test_dpcpp(
            input_dpcpp, target_dpcpp, "none")
        self.assertEqual(output_cpu, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())

        print('sum')
        print("cpu")
        output_cpu, input_cpu = _test_cpu(input_cpu, target_cpu, "sum")
        print("xpu")
        output_dpcpp, input_dpcpp = _test_dpcpp(
            input_dpcpp, target_dpcpp, "sum")
        self.assertEqual(output_cpu, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())

        print('mean')
        print("cpu")
        output_cpu, input_cpu = _test_cpu(input_cpu, target_cpu, "mean")
        print("xpu")
        output_dpcpp, input_dpcpp = _test_dpcpp(
            input_dpcpp, target_dpcpp, "mean")
        self.assertEqual(output_cpu, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())

        print('sum-1024*1024')
        input = torch.randn(1024, 1024)
        target = torch.LongTensor(1024).random_(0, 1024)

        input_cpu = input
        target_cpu = target

        input_dpcpp = input.to("xpu")
        target_dpcpp = target.to("xpu")

        print('cpu')
        output_cpu, input_cpu = _test_cpu(input_cpu, target_cpu, "sum")
        print("xpu")
        output_dpcpp, input_dpcpp = _test_dpcpp(
            input_dpcpp, target_dpcpp, "sum")
        self.assertEqual(output_cpu, output_dpcpp.cpu())
        self.assertEqual(input_cpu.grad, input_dpcpp.grad.cpu())
