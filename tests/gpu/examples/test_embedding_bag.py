import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_embedding_bag(self, dtype=torch.float):
        print("sum cpu")
        embedding_sum = nn.EmbeddingBag(
            10, 3, mode='sum', scale_grad_by_freq=False)
        input = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9], device=cpu_device)
        offsets = torch.LongTensor([0, 4], device=cpu_device)
        weights = torch.Tensor([0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.1, 0.1])
        output = embedding_sum(input, offsets)
        print(output)
        grad_cpu = torch.ones(
            output.shape, device=cpu_device, dtype=torch.float)
        grad_cpu = grad_cpu + grad_cpu
        embedding_sum.zero_grad()

        grad_weight = output.backward(grad_cpu)
        for param in embedding_sum._parameters.values():
            print(param._grad)

        print("sum dpcpp")
        input_dpcpp = input.to(dpcpp_device)
        offsets_dpcpp = offsets.to(dpcpp_device)
        embedding_sum.to(dpcpp_device)
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        weights_dpcpp = weights.to(dpcpp_device)
        output_dpcpp = embedding_sum(input_dpcpp, offsets_dpcpp)
        print(output_dpcpp.to("cpu"))
        self.assertEqual(output, output_dpcpp.to(cpu_device))

        embedding_sum.zero_grad()
        grad_weight = output_dpcpp.backward(grad_dpcpp)
        for param in embedding_sum._parameters.values():
            print(param._grad.to("cpu"))

        print("mean cpu")
        embedding_mean = nn.EmbeddingBag(10, 3, mode='mean')
        output = embedding_mean(input, offsets)
        print(output)

        embedding_mean.zero_grad()
        grad_weight = output.backward(grad_cpu)
        for param in embedding_mean._parameters.values():
            print(param._grad)

        print("mean dpcpp")
        embedding_mean.to(dpcpp_device)
        output_dpcpp = embedding_mean(input_dpcpp, offsets_dpcpp)
        print(output_dpcpp.to("cpu"))

        embedding_mean.zero_grad()
        grad_weight = output_dpcpp.backward(grad_dpcpp)
        for param in embedding_mean._parameters.values():
            print(param._grad.to("cpu"))
        self.assertEqual(output, output_dpcpp.to(cpu_device))

        print("max cpu")
        embedding_max = nn.EmbeddingBag(10, 3, mode='max')
        output = embedding_max(input, offsets)
        print(output)

        embedding_max.zero_grad()
        grad_weight = output.backward(grad_cpu)
        for param in embedding_max._parameters.values():
            print(param._grad)

        print("max dpcpp")
        embedding_max.to(dpcpp_device)
        output_dpcpp = embedding_max(input_dpcpp, offsets_dpcpp)
        print(output_dpcpp.to("cpu"))

        embedding_max.zero_grad()
        grad_weight = output_dpcpp.backward(grad_dpcpp)
        for param in embedding_max._parameters.values():
            print(param._grad.to("cpu"))
        self.assertEqual(output, output_dpcpp.to(cpu_device))
