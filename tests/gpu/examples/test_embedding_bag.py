import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_embedding_bag(self, dtype=torch.float):
        weight_elem = 10
        weight_feature_size = 256

        print("sum cpu")
        embedding_sum = nn.EmbeddingBag(
            weight_elem, weight_feature_size, mode='sum', scale_grad_by_freq=False)
        input = torch.Tensor([1, 2, 4, 5, 4, 3, 2, 9], device=cpu_device).long()
        offsets = torch.Tensor([0, 3, 6], device=cpu_device).long()
        output = embedding_sum(input, offsets)
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

        embedding_sum.zero_grad()
        output_dpcpp.backward(grad_dpcpp)
        for param in embedding_sum._parameters.values():
            grad_weight_dpcpp = copy.deepcopy(param._grad.to("cpu"))
        print('cpu output = ', output)
        print('xpu output = ', output_dpcpp.to("cpu"))
        print('cpu grad = ', grad_weight_cpu)
        print('xpu grad = ', grad_weight_dpcpp.to("cpu"))
        self.assertEqual(output, output_dpcpp.to(cpu_device))
        self.assertEqual(grad_weight_cpu, grad_weight_dpcpp)

        print("mean cpu")
        embedding_mean = nn.EmbeddingBag(weight_elem, weight_feature_size, mode='mean')
        output = embedding_mean(input, offsets)

        embedding_mean.zero_grad()
        output.backward(grad_cpu)
        for param in embedding_mean._parameters.values():
            grad_weight_cpu = copy.deepcopy(param._grad)

        print("mean dpcpp")
        embedding_mean.to(dpcpp_device)
        output_dpcpp = embedding_mean(input_dpcpp, offsets_dpcpp)

        embedding_mean.zero_grad()
        output_dpcpp.backward(grad_dpcpp)
        for param in embedding_mean._parameters.values():
            grad_weight_dpcpp = copy.deepcopy(param._grad.to("cpu"))
        print('cpu output = ', output)
        print('xpu output = ', output_dpcpp.to("cpu"))
        print('cpu grad = ', grad_weight_cpu)
        print('xpu grad = ', grad_weight_dpcpp.to("cpu"))
        self.assertEqual(output, output_dpcpp.to(cpu_device))
        self.assertEqual(grad_weight_cpu, grad_weight_dpcpp)

        print("max cpu")
        embedding_max = nn.EmbeddingBag(weight_elem, weight_feature_size, mode='max')
        output = embedding_max(input, offsets)

        embedding_max.zero_grad()
        output.backward(grad_cpu)
        for param in embedding_max._parameters.values():
            grad_weight_cpu = copy.deepcopy(param._grad)

        print("max dpcpp")
        embedding_max.to(dpcpp_device)
        output_dpcpp = embedding_max(input_dpcpp, offsets_dpcpp)

        embedding_max.zero_grad()
        output_dpcpp.backward(grad_dpcpp)
        for param in embedding_max._parameters.values():
            grad_weight_dpcpp = copy.deepcopy(param._grad.to("cpu"))
        print('cpu output = ', output)
        print('xpu output = ', output_dpcpp.to("cpu"))
        print('cpu grad = ', grad_weight_cpu)
        print('xpu grad = ', grad_weight_dpcpp.to("cpu"))
        self.assertEqual(output, output_dpcpp.to(cpu_device))
        self.assertEqual(grad_weight_cpu, grad_weight_dpcpp.to(cpu_device))

    def test_embedding_bag_large(self, dtype=torch.float):
        weight_elem = 1024
        for weight_feature_size in [127, 128, 512, 5000]:
            print("weight_feature_size = ", weight_feature_size)
            print("sum cpu")
            embedding_sum = nn.EmbeddingBag(
                weight_elem, weight_feature_size, mode='sum', scale_grad_by_freq=False)
            input = torch.Tensor([99, 299, 499, 599, 399, 799, 199, 999], device=cpu_device).long()
            offsets = torch.Tensor([0, 3, 4, 6, 7], device=cpu_device).long()
            output = embedding_sum(input, offsets)
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

            embedding_sum.zero_grad()
            output_dpcpp.backward(grad_dpcpp)
            for param in embedding_sum._parameters.values():
                grad_weight_dpcpp = copy.deepcopy(param._grad.to("cpu"))
            print('cpu output = ', output)
            print('xpu output = ', output_dpcpp.to("cpu"))
            print('cpu grad = ', grad_weight_cpu)
            print('xpu grad = ', grad_weight_dpcpp.to("cpu"))
            self.assertEqual(output, output_dpcpp.to(cpu_device))
            self.assertEqual(grad_weight_cpu, grad_weight_dpcpp)


            print("mean cpu")
            embedding_mean = nn.EmbeddingBag(weight_elem, weight_feature_size, mode='mean')
            output = embedding_mean(input, offsets)

            embedding_mean.zero_grad()
            output.backward(grad_cpu)
            for param in embedding_mean._parameters.values():
                grad_weight_cpu = copy.deepcopy(param._grad)

            print("mean dpcpp")
            embedding_mean.to(dpcpp_device)
            output_dpcpp = embedding_mean(input_dpcpp, offsets_dpcpp)

            embedding_mean.zero_grad()
            output_dpcpp.backward(grad_dpcpp)
            for param in embedding_mean._parameters.values():
                grad_weight_dpcpp = copy.deepcopy(param._grad.to("cpu"))
            print('cpu output = ', output)
            print('xpu output = ', output_dpcpp.to("cpu"))
            print('cpu grad = ', grad_weight_cpu)
            print('xpu grad = ', grad_weight_dpcpp.to("cpu"))
            self.assertEqual(output, output_dpcpp.to(cpu_device))
            self.assertEqual(grad_weight_cpu, grad_weight_dpcpp)


            print("max cpu")
            embedding_max = nn.EmbeddingBag(weight_elem, weight_feature_size, mode='max')
            output = embedding_max(input, offsets)

            embedding_max.zero_grad()
            output.backward(grad_cpu)
            for param in embedding_max._parameters.values():
                grad_weight_cpu = copy.deepcopy(param._grad)

            print("max dpcpp")
            embedding_max.to(dpcpp_device)
            output_dpcpp = embedding_max(input_dpcpp, offsets_dpcpp)

            embedding_max.zero_grad()
            output_dpcpp.backward(grad_dpcpp)
            for param in embedding_max._parameters.values():
                grad_weight_dpcpp = copy.deepcopy(param._grad.to("cpu"))
            print('cpu output = ', output)
            print('xpu output = ', output_dpcpp.to("cpu"))
            print('cpu grad = ', grad_weight_cpu)
            print('xpu grad = ', grad_weight_dpcpp.to("cpu"))
            self.assertEqual(output, output_dpcpp.to(cpu_device))
            self.assertEqual(grad_weight_cpu, grad_weight_dpcpp)

    def test_embedding_bag_DLRM(self, dtype=torch.float16):
        weight_elem = 1024
        weight_feature_size = 128
        print("weight_feature_size = ", weight_feature_size)
        print("sum cpu")
        embedding_sum = nn.EmbeddingBag(
            weight_elem, weight_feature_size, mode='sum', scale_grad_by_freq=False)
        input = torch.Tensor([111, 222, 444, 999, 555, 333, 0, 777, 666, 888], device=cpu_device).long()
        offsets = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], device=cpu_device).long()
        output = embedding_sum(input, offsets)

        print("sum dpcpp")
        input_dpcpp = input.to(dpcpp_device)
        offsets_dpcpp = offsets.to(dpcpp_device)
        embedding_sum.to(device=dpcpp_device, dtype=dtype)
        output_dpcpp = embedding_sum(input_dpcpp, offsets_dpcpp)

        print('sum cpu = ', output)
        print('sum dpcpp = ', output_dpcpp.to(cpu_device))
        self.assertEqual(output, output_dpcpp.to(device=cpu_device).float(), atol=1e-2, rtol=0)
