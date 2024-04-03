import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

sycl_device = torch.device("xpu")
device = sycl_device

class TestNNMethod(TestCase):
    def test_dropout_1(self):
        p = 0.2
        # cover the code path without tailing
        input = torch.randn(1, 512, 1024)
        input = input.to(device).fill_(1 - p)  # input valuses are 0.8

        module = nn.Dropout(p)
        input_var = input.clone().requires_grad_()
        output = module(
            input_var
        )  # output values are 0.0 and 1.0 (input_value * 1/(1-p))
        self.assertLess(abs(output.cpu().data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertLess(abs(input_var.grad.cpu().data.mean() - (1 - p)), 0.05)

        # cover the code path with tailing
        input = torch.randn(1, 512, 1024)
        input = input.to(device).fill_(1 - p)  # input valuses are 0.8

        module = nn.Dropout(p)
        input_var = input.clone().requires_grad_()
        output = module(
            input_var
        )  # output values are 0.0 and 1.0 (input_value * 1/(1-p))
        
        self.assertLess(abs(output.cpu().data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertLess(abs(input_var.grad.cpu().data.mean() - (1 - p)), 0.05)

        # cover the code path with non-contiguous input
        input_orig = torch.randn(1, 512, 1024)
        input = input_orig.transpose(1, 0)
        input = input.to(device).fill_(1 - p)  # input valuses are 0.8

        module = nn.Dropout(p)
        input_var = input.clone().requires_grad_()
        output = module(
            input_var
        )  # output values are 0.0 and 1.0 (input_value * 1/(1-p))
        
        self.assertLess(abs(output.cpu().data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertLess(abs(input_var.grad.cpu().data.mean() - (1 - p)), 0.05)

    def test_dropout_2(self):
        p = 0.2
        # cover the code path without tailing
        input = torch.randn(1, 16, 512, 512)
        input = input.to(device).fill_(1 - p)  # input valuses are 0.8

        module = nn.Dropout(p)
        input_var = input.clone().requires_grad_()
        output = module(
            input_var
        )  # output values are 0.0 and 1.0 (input_value * 1/(1-p))
        self.assertLess(abs(output.cpu().data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertLess(abs(input_var.grad.cpu().data.mean() - (1 - p)), 0.05)

        # cover the code path with tailing
        input = torch.randn(1, 16, 512, 512)
        input = input.to(device).fill_(1 - p)  # input valuses are 0.8

        module = nn.Dropout(p)
        input_var = input.clone().requires_grad_()
        output = module(
            input_var
        )  # output values are 0.0 and 1.0 (input_value * 1/(1-p))
        
        self.assertLess(abs(output.cpu().data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertLess(abs(input_var.grad.cpu().data.mean() - (1 - p)), 0.05)

        # cover the code path with non-contiguous input
        input_orig = torch.randn(1, 16, 512, 512)
        input = input_orig.transpose(1, 0)
        input = input.to(device).fill_(1 - p)  # input valuses are 0.8

        module = nn.Dropout(p)
        input_var = input.clone().requires_grad_()
        output = module(
            input_var
        )  # output values are 0.0 and 1.0 (input_value * 1/(1-p))
        
        self.assertLess(abs(output.cpu().data.mean() - (1 - p)), 0.05)
        output.backward(input)
        self.assertLess(abs(input_var.grad.cpu().data.mean() - (1 - p)), 0.05)
