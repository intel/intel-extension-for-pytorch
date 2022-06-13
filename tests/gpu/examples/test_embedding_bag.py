import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import pytest

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")
ATOL = 1e-5
RTOL = 1e-5

class TestTorchMethod(TestCase):
    def test_embedding_bag_all(self, dtype=torch.float32):
        weight_elem = 56
        for weight_feature_size in [1, 127, 128]:
            for mode in ['sum', 'mean', 'max']:
                for include_last_offset in [False, True]:
                    for padding_idx in [None, 29, 10]:
                        embedding = nn.EmbeddingBag(weight_elem, weight_feature_size, mode=mode, scale_grad_by_freq=False, include_last_offset=include_last_offset, padding_idx=padding_idx)
                        input = torch.Tensor([9, 29, 49, 39, 19, 29, 19, 9, 0], device=cpu_device).long()
                        offsets = torch.Tensor([0, 1, 2, 4, 7, 9], device=cpu_device).long()
                        output = embedding(input, offsets)
                        grad_cpu = torch.randn(output.shape, device=cpu_device, dtype=torch.float)
                        embedding.zero_grad()

                        output.backward(grad_cpu)
                        for param in embedding._parameters.values():
                            grad_weight_cpu = copy.deepcopy(param._grad)

                        input_xpu = input.to(xpu_device)
                        offsets_xpu = offsets.to(xpu_device)
                        embedding_xpu = embedding.to(xpu_device, dtype=dtype)
                        grad_xpu = grad_cpu.to(xpu_device, dtype=dtype)
                        print('weight_elem = ', weight_elem, '. weight_feature_size = ', weight_feature_size)
                        print('mode = ', mode, '. dtype = ', dtype, '. include_last_offset = ', include_last_offset, '. padding_idx = ', padding_idx)
                        output_xpu = embedding_xpu(input_xpu, offsets_xpu)

                        embedding_xpu.zero_grad()
                        output_xpu.backward(grad_xpu)
                        for param in embedding_xpu._parameters.values():
                            grad_weight_xpu = copy.deepcopy(param._grad.to("cpu"))

                        print('output = \n', output)
                        print('output_xpu = \n', output_xpu.cpu())
                        self.assertEqual(output, output_xpu.to(cpu_device).float(), atol=ATOL, rtol=RTOL)
                        print('grad_weight_cpu = \n', grad_weight_cpu)
                        print('grad_weight_xpu = \n', grad_weight_xpu.cpu())
                        self.assertEqual(grad_weight_cpu, grad_weight_xpu.to(cpu_device).float(), atol=ATOL, rtol=RTOL)
