import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import copy
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif("not torch_ipex._onedpl_is_enabled()")
    def test_embedding_bag(self, dtype=torch.float):
        print("sum cpu")
        embedding_sum = nn.EmbeddingBag(
            10, 3, mode='sum', scale_grad_by_freq=False)
        input = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9], device=cpu_device)
        offsets = torch.LongTensor([0, 4], device=cpu_device)
        output = embedding_sum(input, offsets)
        print(output)
        grad_cpu = torch.randn(output.shape, device=cpu_device, dtype=torch.float)
        embedding_sum.zero_grad()

        output.backward(grad_cpu)
        for param in embedding_sum._parameters.values():
            grad_weight_cpu = copy.deepcopy(param._grad)

        print("sum dpcpp")
        input_dpcpp = input.to(dpcpp_device)
        offsets_dpcpp = offsets.to(dpcpp_device)
        embedding_sum.to(dpcpp_device)
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        output_dpcpp = embedding_sum(input_dpcpp, offsets_dpcpp)
        print(output_dpcpp.to("cpu"))
        
        embedding_sum.zero_grad()
        output_dpcpp.backward(grad_dpcpp)
        for param in embedding_sum._parameters.values():
            grad_weight_dpcpp = copy.deepcopy(param._grad.to("cpu"))
        print(grad_weight_cpu)
        print(grad_weight_dpcpp)
        self.assertEqual(output, output_dpcpp.to(cpu_device))
        self.assertEqual(grad_weight_cpu, grad_weight_dpcpp)


        print("mean cpu")
        embedding_mean = nn.EmbeddingBag(10, 3, mode='mean')
        output = embedding_mean(input, offsets)
        print(output)

        embedding_mean.zero_grad()
        output.backward(grad_cpu)
        for param in embedding_mean._parameters.values():
            grad_weight_cpu = copy.deepcopy(param._grad)

        print("mean dpcpp")
        embedding_mean.to(dpcpp_device)
        output_dpcpp = embedding_mean(input_dpcpp, offsets_dpcpp)
        print(output_dpcpp.to("cpu"))

        embedding_mean.zero_grad()
        output_dpcpp.backward(grad_dpcpp)
        for param in embedding_mean._parameters.values():
            grad_weight_dpcpp = copy.deepcopy(param._grad.to("cpu"))
        print(grad_weight_cpu)
        print(grad_weight_dpcpp)
        self.assertEqual(output, output_dpcpp.to(cpu_device))
        self.assertEqual(grad_weight_cpu, grad_weight_dpcpp)


        print("max cpu")
        embedding_max = nn.EmbeddingBag(10, 3, mode='max')
        output = embedding_max(input, offsets)
        print(output)

        embedding_max.zero_grad()
        output.backward(grad_cpu)
        for param in embedding_max._parameters.values():
            grad_weight_cpu = copy.deepcopy(param._grad)

        print("max dpcpp")
        embedding_max.to(dpcpp_device)
        output_dpcpp = embedding_max(input_dpcpp, offsets_dpcpp)
        print(output_dpcpp.to("cpu"))

        embedding_max.zero_grad()
        output_dpcpp.backward(grad_dpcpp)
        for param in embedding_max._parameters.values():
            grad_weight_dpcpp = copy.deepcopy(param._grad.to("cpu"))
        print(grad_weight_cpu)
        print(grad_weight_dpcpp)
        self.assertEqual(output, output_dpcpp.to(cpu_device))
        self.assertEqual(grad_weight_cpu, grad_weight_dpcpp)
