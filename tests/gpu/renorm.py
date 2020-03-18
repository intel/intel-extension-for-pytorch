import torch

import torch_ipex
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x_cpu = torch.ones(3, 3, device=cpu_device)

x_cpu[1].fill_(2)
x_cpu[2].fill_(3)
x_sycl = x_cpu.to(sycl_device)

renorm = torch.renorm(x_cpu, 1, 1, 5)

print("torch.renorm(x_cpu, 1, 1, 5)", renorm)

renorm = torch.renorm(x_sycl, 1, 1, 5)

print("torch.renorm(x_sycl, 1, 1, 5)", renorm.cpu())

renorm = torch.renorm(x_cpu, 1, 0, 5)

print("torch.renorm(x_cpu, 1, 0, 5)", renorm)

renorm = torch.renorm(x_sycl, 1, 0, 5)

print("torch.renorm(x_sycl, 1, 0, 5)", renorm.cpu())
