import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_pdist(self, dtype=torch.float):
        # cpu
        input = torch.randn(10, 512)
        input.requires_grad_(True)
        output = nn.functional.pdist(input)
        print(output.size())
        print(output)
        output.backward(torch.ones(output.size(), device=cpu_device))
        print(input.grad)
        input.requires_grad_(False)
        # dpcpp
        input1_dpcpp = input.to(dpcpp_device)
        input1_dpcpp.requires_grad_(True)
        output_dpcpp = nn.functional.pdist(input1_dpcpp)
        print(output_dpcpp.size())
        print(output_dpcpp.cpu())
        output_dpcpp.backward(torch.ones(
            output_dpcpp.size(), device=dpcpp_device))
        print(input1_dpcpp.grad.cpu())
        self.assertEqual(output, output_dpcpp.cpu())
        self.assertEqual(input.grad, input1_dpcpp.grad.cpu())
