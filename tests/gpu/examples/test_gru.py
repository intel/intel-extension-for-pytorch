import copy
import pytest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

cpu_device = torch.device('cpu')
xpu_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_gru(self, dtype=torch.float):
        rnn = nn.GRU(2, 3, 2, bias=True, bidirectional=False)
        rnn_xpu = copy.deepcopy(rnn).to("xpu")
        input = torch.randn(2, 3, 2)
        h0 = torch.randn(2, 3, 3)
        input_xpu = input.to("xpu")
        h0_xpu = h0.to("xpu")
        grad_output = torch.randn(2, 3, 3)
        grad_output_xpu = grad_output.to("xpu")

        input.requires_grad = True
        h0.requires_grad = True
        output, hn = rnn(input, h0)
        print(output)
        grad_output.requires_grad = True
        output.backward(grad_output)
        print(input.grad)
        param_grad = []
        for param in rnn._parameters.values():
            param_grad.append(param._grad.clone())

        input_xpu.requires_grad = True
        h0_xpu.requires_grad = True
        output_xpu, hn_xpu = rnn_xpu(input_xpu, h0_xpu)
        print(output_xpu.cpu())
        grad_output_xpu.requires_grad = True
        output_xpu.backward(grad_output_xpu)
        print(input_xpu.grad.cpu())
        param_grad_xpu = []
        for param in rnn_xpu._parameters.values():
            param_grad_xpu.append(param._grad.clone())

        self.assertEqual(output, output_xpu.cpu())
        self.assertEqual(h0, h0.cpu())
        self.assertEqual(input.grad, input_xpu.grad.cpu())
        self.assertEqual(h0.grad, h0_xpu.grad.cpu())
        for i in range(len(param_grad)):
            self.assertEqual(param_grad[i], param_grad_xpu[i].cpu())

    def test_gru_bf16(self, dtype=torch.bfloat16):
        rnn = nn.GRU(379, 681, num_layers=3, batch_first=True)
        rnn_xpu = copy.deepcopy(rnn).to("xpu").to(torch.bfloat16)
        input = torch.randn(512, 63, 379)
        h0 = torch.randn(3, 512, 681)
        input_xpu = input.to("xpu").to(torch.bfloat16)
        h0_xpu = h0.to("xpu").to(torch.bfloat16)
        grad_output = torch.randn(512, 63, 681)
        grad_output_xpu = grad_output.to("xpu").to(torch.bfloat16)

        input.requires_grad = True
        h0.requires_grad = True
        output, hn = rnn(input, h0)
        print(output)
        grad_output.requires_grad = True
        output.backward(grad_output)
        print(input.grad)
        param_grad = []
        for param in rnn._parameters.values():
            param_grad.append(param._grad.clone())

        input_xpu.requires_grad = True
        h0_xpu.requires_grad = True
        output_xpu, hn_xpu = rnn_xpu(input_xpu, h0_xpu)
        print(output_xpu.cpu())
        grad_output_xpu.requires_grad = True
        output_xpu.backward(grad_output_xpu)
        print(input_xpu.grad)
        param_grad_xpu = []
        for param in rnn_xpu._parameters.values():
            param_grad_xpu.append(param._grad.clone())

        self.assertEqual(output, output_xpu.float().cpu(), atol=3e-2, rtol=8e-3)
        self.assertEqual(h0, h0.float().cpu())
        self.assertEqual(input.grad, input_xpu.grad.float().cpu(), atol=3e-2, rtol=8e-3)
        self.assertEqual(h0.grad, h0_xpu.grad.float().cpu(), atol=3e-2, rtol=8e-3)
