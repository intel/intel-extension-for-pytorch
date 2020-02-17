from __future__ import print_function
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")
x_cpu = torch.tensor([[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]], requires_grad=True, device = cpu_device)
y_cpu_output = torch.tensor([[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]], requires_grad=True, device = cpu_device)

#x_cpu = torch.randn([2,500], device=cpu_device, dtype=dtype)
#x_cpu = torch.tensor(x_cpu, requires_grad=True)
y = F.log_softmax(x_cpu, 1)
# y.backward(y_cpu_output)

print("x:", x_cpu)
print("log_softmax:", y)
# print("log_softmax x_cpu_grad = ", x_cpu.grad)

# y = F.softmax(x_cpu, 1)
# y.backward(y_cpu_output)
# 
# print("softmax:", y)
# print("softmax x_cpu_grad = ", x_cpu.grad)

x_sycl = torch.tensor([[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]], requires_grad=True, device = sycl_device)
y_sycl_output = torch.tensor([[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]], requires_grad=True, device = sycl_device)

y_sycl = F.log_softmax(x_sycl, 1)
# y_sycl.backward(y_sycl_output)

print("x_sycl:", x_sycl.cpu())
print("log_softmax sycl:", y_sycl.cpu())
# print("log_softmax x_sycl_grad = ", x_sycl.grad.cpu())

# y_sycl = F.softmax(x_sycl, 1)
# y_sycl.backward(y_sycl_output)
# 
# print("softmax sycl:", y_sycl.cpu())
# print("softmax x_sycl_grad = ", x_sycl.grad.cpu())
