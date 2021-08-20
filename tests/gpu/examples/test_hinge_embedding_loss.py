import torch
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_hinge_embedding(self, dtype=torch.float):

        input = torch.randn(3, 5)
        target = torch.ones(3, 5)

        input_cpu = input
        target_cpu = target

        input_dpcpp = input.to(dpcpp_device)
        target_dpcpp = target.to(dpcpp_device)

        def _test_cpu(input, target, reduc):
            loss = nn.HingeEmbeddingLoss(reduction=reduc)
            input.requires_grad = True
            output = loss(input, target)
            print(output)
            if(reduc == "none"):
                output.backward(torch.ones_like(input, dtype=torch.float))
            else:
                output.backward(torch.tensor((1.0), dtype=torch.float))
            print(input.grad)
            input.grad.zero_()
            return input, output

        def _test_dpcpp(input, target, reduc):
            loss = nn.HingeEmbeddingLoss(reduction=reduc)
            input.requires_grad = True
            output = loss(input, target)
            print(output.cpu())
            if(reduc == "none"):
                output.backward(torch.ones_like(
                    input, dtype=torch.float).to(dpcpp_device))
            else:
                output.backward(torch.tensor(
                    (1.0), dtype=torch.float).to(dpcpp_device))
            print(input.grad.cpu())
            input.grad.zero_()
            return input, output

        print('none')
        print("cpu")
        input_cpu, output_cpu = _test_cpu(input_cpu, target_cpu, "none")
        print("xpu")
        input_dpcpp, output_dpcpp = _test_dpcpp(
            input_dpcpp, target_dpcpp, "none")
        self.assertEqual(input_cpu, input_dpcpp.to(cpu_device))
        self.assertEqual(output_cpu, output_dpcpp.to(cpu_device))

        print('sum')
        print("cpu")
        input_cpu, output_cpu = _test_cpu(input_cpu, target_cpu, "sum")
        print("xpu")
        input_dpcpp, output_dpcpp = _test_dpcpp(
            input_dpcpp, target_dpcpp, "sum")
        self.assertEqual(input_cpu, input_dpcpp.to(cpu_device))
        self.assertEqual(output_cpu, output_dpcpp.to(cpu_device))

        print('mean')
        print("cpu")
        input_cpu, output_cpu = _test_cpu(input_cpu, target_cpu, "mean")
        print("xpu")
        input_dpcpp, output_dpcpp = _test_dpcpp(
            input_dpcpp, target_dpcpp, "mean")
        self.assertEqual(input_cpu, input_dpcpp.to(cpu_device))
        self.assertEqual(output_cpu, output_dpcpp.to(cpu_device))
