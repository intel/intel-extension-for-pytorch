import torch
import torch.nn as nn
import ipex
import copy

conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
conv_xpu = copy.deepcopy(conv).bfloat16().to("xpu")
upsample = nn.Upsample(scale_factor=2, mode="nearest")
upsample_xpu = copy.deepcopy(upsample).bfloat16().to("xpu")
conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
conv2_xpu = copy.deepcopy(conv2).bfloat16().to("xpu")
upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
upsample2_xpu = copy.deepcopy(upsample2).bfloat16().to("xpu")


input = torch.randn([64, 64, 64, 64])
input_xpu = copy.deepcopy(input).bfloat16().to("xpu")
input.requires_grad_(True)
input_xpu.requires_grad_(True)

output = upsample2(conv2(upsample(conv(input))))

#print(linear.weight.data.cpu())

grad = torch.randn([64, 64, 256, 256])
grad_xpu = copy.deepcopy(grad).bfloat16().to("xpu")
output.backward(grad)
output_xpu = upsample2_xpu(conv2_xpu(upsample_xpu(conv_xpu(input_xpu))))
output_xpu.backward(grad_xpu)
print(output_xpu.cpu()-output)
print(input_xpu.grad.cpu()-input.grad)
#print(input.grad)
#print(input_xpu.grad.cpu())
