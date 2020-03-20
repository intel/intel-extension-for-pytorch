import torch
import torch.nn as nn

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

print('none')
loss = nn.L1Loss(reduction="none")
input = torch.randn(3, 5, requires_grad = True)
target = torch.randn(3, 5)

print("cpu")
input_cpu = input
target_cpu = target
output_cpu = loss(input_cpu, target_cpu)
print(output_cpu)
output_cpu.backward(torch.ones_like(target_cpu, dtype=torch.float))
print(input_cpu.grad)
input_cpu.grad.zero_()

print("sycl")
input_sycl = input
target_sycl = target
output_sycl = loss(input_sycl.to("dpcpp"), target_sycl.to("dpcpp"))
print(output_sycl.cpu())
output_sycl.backward(torch.ones_like(target_sycl, dtype=torch.float, device=sycl_device))
print(input_sycl.grad.cpu())
input_sycl.grad.zero_()

print('sum')
loss = nn.L1Loss(reduction="sum")

print("cpu")
input_cpu = input
target_cpu = target
output_cpu = loss(input_cpu, target_cpu)
print(output_cpu)
output_cpu.backward(torch.tensor((2.0), dtype=torch.float))
print(input_cpu.grad)
input_cpu.grad.zero_()

print("sycl")
input_sycl = input
target_sycl = target
output_sycl = loss(input_sycl.to("dpcpp"), target_sycl.to("dpcpp"))
print(output_sycl.cpu())
output_sycl.backward(torch.tensor((2.0), dtype=torch.float, device=sycl_device))
print(input_sycl.grad.cpu())
input_sycl.grad.zero_()

print('mean')
loss = nn.L1Loss(reduction="mean")

print("cpu")
input_cpu = input
target_cpu = target
output_cpu = loss(input_cpu, target_cpu)
print(output_cpu)
output_cpu.backward(torch.tensor((2.0), dtype=torch.float))
print(input_cpu.grad)
input_cpu.grad.zero_()

print("sycl")
input_sycl = input
target_sycl = target
output_sycl = loss(input_sycl.to("dpcpp"), target_sycl.to("dpcpp"))
print(output_sycl.cpu())
output_sycl.backward(torch.tensor((2.0), dtype=torch.float, device=sycl_device))
print(input_sycl.grad.cpu())
input_sycl.grad.zero_()
