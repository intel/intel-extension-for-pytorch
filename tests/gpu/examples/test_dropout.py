import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import ipex
import pytest

sycl_device = torch.device("xpu")
device = sycl_device


class TestNNMethod(TestCase):
    def test_dropout(self):
        p = 0.2
        input = torch.Tensor(1000)
        input = input.to(device).fill_(1 - p)  # input valuses are 0.8

        module = nn.Dropout(p)
        input_var = input.clone().requires_grad_()
        output = module(input_var)  # output values are 0.0 and 1.0 (input_value * 1/(1-p))
        # for Bernoulli distribution:
        # x=0, p
        # x=1, (1-p)
        # Thus the mean value of output tensor is (1-p)
        self.assertLess(abs(output.cpu().data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertLess(abs(input_var.grad.cpu().data.mean() - (1 - p)), 0.05)
