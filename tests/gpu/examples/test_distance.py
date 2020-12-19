import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_distance(self, dtype=torch.float):
        pdist = nn.PairwiseDistance(p=2)
        input1 = torch.randn(100, 128, device=cpu_device,
                             dtype=dtype, requires_grad=True)
        input2 = torch.randn(100, 128, device=cpu_device,
                             dtype=dtype, requires_grad=True)
        output = pdist(input1, input2)
        print(output)
        pdist_dpcpp = pdist.to(dpcpp_device)
        input1_dpcpp = torch.randn(
            100, 128, device=dpcpp_device, dtype=dtype, requires_grad=True)
        input2_dpcpp = torch.randn(
            100, 128, device=dpcpp_device, dtype=dtype, requires_grad=True)
        output_dpcpp = pdist_dpcpp(input1, input2)
        print(output_dpcpp.to(cpu_device))
        self.assertEqual(output, output_dpcpp.to(cpu_device))

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        input1 = torch.randn(100, 128, device=cpu_device,
                             dtype=dtype, requires_grad=True)
        input2 = torch.randn(100, 128, device=cpu_device,
                             dtype=dtype, requires_grad=True)
        output = cos(input1, input2)
        print(output)
        cos_dpcpp = cos.to(dpcpp_device)
        input1_dpcpp = torch.randn(
            100, 128, device=dpcpp_device, dtype=dtype, requires_grad=True)
        input2_dpcpp = torch.randn(
            100, 128, device=dpcpp_device, dtype=dtype, requires_grad=True)
        output_dpcpp = cos_dpcpp(input1, input2)
        print(output_dpcpp.to(cpu_device))
        self.assertEqual(output, output_dpcpp.to(cpu_device))
