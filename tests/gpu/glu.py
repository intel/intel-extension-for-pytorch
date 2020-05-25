import torch
from torch import nn

import torch_ipex


input_cpu = torch.randn(4, 6)
input_sycl = input_cpu.to("dpcpp")
m = nn.GLU()

print("cpu")
input_cpu.requires_grad = True
output_cpu = m(input_cpu)
print("output: ", output_cpu)
output_cpu.backward(torch.ones_like(output_cpu))
print("input.grad: ", input_cpu.grad)
input_cpu.grad.zero_()

print("sycl")
input_sycl.requires_grad = True
output_sycl = m(input_sycl)
print("output: ", output_sycl.cpu())
output_sycl.backward(torch.ones_like(output_sycl).to("dpcpp"))
print("input.grad: ", input_sycl.grad.cpu())
input_sycl.grad.zero_()
