import torch
import torch.nn as nn
import ipex
import copy

conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
conv_xpu = copy.deepcopy(conv).bfloat16().to("xpu")
relu = nn.ReLU()
relu_xpu = copy.deepcopy(relu).to("xpu")


input = torch.randn([32, 32, 64, 64])
input_xpu = copy.deepcopy(input).bfloat16().to("xpu")
input.requires_grad_(True)
input_xpu.requires_grad_(True)

output = relu(conv(input))
print(output)
grad = torch.randn([32, 32, 64, 64])
output.backward(grad)
print(input.grad)

output_xpu = relu_xpu(conv_xpu(input_xpu))
print(output_xpu.cpu())
grad_xpu = copy.deepcopy(grad).bfloat16().to("xpu")
output_xpu.backward(grad_xpu)
print(input_xpu.grad.cpu())
