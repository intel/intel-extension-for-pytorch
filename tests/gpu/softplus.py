from __future__ import print_function
import torch
import torch_ipex
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.float
cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")
x_cpu = torch.tensor([[0.5, 1.5, 0.1], [2.2, 1.2, 1.7]], requires_grad=True, device = cpu_device)
y_cpu_output = torch.tensor([[0.5, 1.5, 0.1], [2.2, 1.2, 1.7]], requires_grad=True, device = cpu_device)

y = F.softplus(x_cpu, 3, 5)
y.backward(y_cpu_output)

print("CPU Result:")
print("x:", x_cpu)
print("softplus:", y)
print("softplus x_cpu_grad = ", x_cpu.grad)

x_dpcpp = torch.tensor([[0.5, 1.5, 0.1], [2.2, 1.2, 1.7]], requires_grad=True, device = dpcpp_device)
y_dpcpp_output = torch.tensor([[0.5, 1.5, 0.1], [2.2, 1.2, 1.7]], requires_grad=True, device = dpcpp_device)

y_dpcpp = F.softplus(x_dpcpp, 3, 5)
y_dpcpp.backward(y_dpcpp_output)

print("SYCL Result:")
print("x_dpcpp:", x_dpcpp.cpu())
print("softplus dpcpp:", y_dpcpp.cpu())
print("softplus x_dpcpp_grad = ", x_dpcpp.grad.cpu())
