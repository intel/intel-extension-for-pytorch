import copy

import torch
import torch.nn as nn

import ipex

linear = nn.Linear(1024, 512, bias=False)
linear_xpu = copy.deepcopy(linear).bfloat16().to("xpu")
softmax = nn.Softmax(dim=1)
softmax_xpu = copy.deepcopy(softmax).bfloat16().to("xpu")


input = torch.randn([1024, 1024])
input_xpu = copy.deepcopy(input).bfloat16().to("xpu")
input.requires_grad_(True)
input_xpu.requires_grad_(True)

output_xpu = softmax_xpu(linear_xpu(input_xpu))
print(output_xpu.cpu())
# print(linear.weight.data.cpu())

grad = torch.randn([1024, 512])
grad_xpu = copy.deepcopy(grad).bfloat16().to("xpu")
output_xpu.backward(grad_xpu)
print(input_xpu.grad.cpu())

output = softmax(linear(input))
print(output)
output.backward(grad)
print(input.grad.cpu())
