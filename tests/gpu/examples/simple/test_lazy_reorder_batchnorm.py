import torch
import torch.nn as nn
import torch_ipex
import copy

conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
conv_xpu = copy.deepcopy(conv).bfloat16().to("xpu")
bn = nn.BatchNorm2d(64)
bn_xpu = copy.deepcopy(bn).bfloat16().to("xpu")


input = torch.randn([64, 64, 64, 64])
input_xpu = copy.deepcopy(input).bfloat16().to("xpu")
input.requires_grad_(True)
input_xpu.requires_grad_(True)

output = bn(conv(input))
print(output)
#print(linear.weight.data.cpu())

grad = torch.randn([64, 64, 64, 64])
output.backward(grad)
print(input.grad)
output_xpu = bn_xpu(conv_xpu(input_xpu))
print(output_xpu.cpu())
grad_xpu = copy.deepcopy(grad).bfloat16().to("xpu")
output_xpu.backward(grad_xpu)
print(input_xpu.grad.cpu())
# print(input_xpu.grad.cpu())


