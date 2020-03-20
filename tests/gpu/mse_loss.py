import torch
import torch.nn as nn

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

print('none')
loss = nn.MSELoss(reduction="none")
input = torch.randn(3, 5)
input_sycl = input.to("dpcpp")
target = torch.randn(3, 5)

print("cpu")
input_cpu = input
target_cpu = target
input_cpu.requires_grad = True
output_cpu = loss(input_cpu, target_cpu)
print(output_cpu)
output_cpu.backward(torch.ones_like(target_cpu, dtype=torch.float))
print(input_cpu.grad)
input_cpu.grad.zero_()

print("sycl")

target_sycl = target.to("dpcpp")
input_sycl.requires_grad = True
output_sycl = loss(input_sycl, target_sycl)
print(output_sycl.cpu())
output_sycl.backward(torch.ones_like(target, dtype=torch.float, device=sycl_device))
print(input_sycl.grad.cpu())
input_sycl.grad.zero_()

print("sum")
loss = nn.MSELoss(reduction="sum")

print("cpu")
input_cpu = input
target_cpu = target
output_cpu = loss(input_cpu, target_cpu)
print(output_cpu)
output_cpu.backward(torch.tensor((2.0), dtype=torch.float))
print(input_cpu.grad)
input_cpu.grad.zero_()

print("sycl")
target_sycl = target.to("dpcpp")
input_sycl.requires_grad = True
output_sycl = loss(input_sycl, target_sycl)
print(output_sycl.cpu())
output_sycl.backward(torch.tensor((2.0), dtype=torch.float, device=sycl_device))
print(input_sycl.grad.cpu())
input_sycl.grad.zero_()

print("mean")
loss = nn.MSELoss(reduction="mean")

print("cpu")
input_cpu = input
target_cpu = target
output_cpu = loss(input_cpu, target_cpu)
print(output_cpu)
output_cpu.backward(torch.tensor((2.0), dtype=torch.float))
print(input_cpu.grad)
input_cpu.grad.zero_()

print("sycl")
target_sycl = target.to("dpcpp")
input_sycl.requires_grad = True
output_sycl = loss(input_sycl, target_sycl)
print(output_sycl.cpu())
output_sycl.backward(torch.tensor((2.0), dtype=torch.float, device=sycl_device))
print(input_sycl.grad.cpu())
input_sycl.grad.zero_()
