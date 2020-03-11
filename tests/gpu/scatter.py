import torch

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x_cpu = torch.rand(2, 5)
x_sycl = x_cpu.to(sycl_device)

y_cpu = torch.zeros(3, 5).scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x_cpu)

print("y_cpu", y_cpu)

y_sycl = torch.zeros(3, 5, device=sycl_device).scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], device=sycl_device), x_sycl)

print("y_sycl", y_sycl.cpu())

z_cpu = torch.zeros(2, 4).scatter_(1, torch.tensor([[2], [3]]), 1.23)

print("z_cpu", z_cpu)

z_sycl = torch.zeros(2, 4, device=sycl_device).scatter_(1, torch.tensor([[2], [3]], device=sycl_device), 1.23)

print("z_sycl", z_sycl.cpu())
