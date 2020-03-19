import numpy
import torch
import torch.nn as nn

import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x_cpu = torch.ones([8, 8, 24, 24], device=cpu_device, dtype=dtype)
grad_cpu = torch.ones([8, 8, 24, 24], device=cpu_device, dtype=dtype)

avg_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)

#cpu
x_cpu.requires_grad_(True)
y_cpu = avg_pool(x_cpu)
print("y_cpu", y_cpu)
y_cpu.backward(torch.ones([8, 8, 24, 24], device=cpu_device))
print("y_cpu backward", x_cpu.grad)


x_sycl = torch.ones([8, 8, 24, 24], device=sycl_device, dtype=dtype)
x_sycl.requires_grad_(True)
y_sycl = avg_pool(x_sycl)

print("y_sycl", y_sycl.cpu())

# grad_sycl = grad_cpu.to("sycl")
y_sycl.backward(torch.ones([8, 8, 24, 24], device=sycl_device))
print("y_sycl backward", x_sycl.grad.cpu())

