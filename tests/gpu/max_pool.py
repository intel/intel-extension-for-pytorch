import numpy
import torch
import torch.nn as nn
import torch_ipex

dtype = torch.float
dtype1 = torch.long
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x_cpu = torch.ones([2, 2, 3, 3], device=cpu_device, dtype=dtype)
grad_cpu = torch.ones([2, 2, 3, 3], device=cpu_device, dtype=dtype)

conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=True)
max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, return_indices=True)

y_cpu = conv1(x_cpu)
y_cpu = max_pool(y_cpu)
print("y_cpu", y_cpu[0])

conv1.to("dpcpp")
x_sycl = x_cpu.to("dpcpp")
y_sycl = conv1(x_sycl)
y_sycl = max_pool(y_sycl)
print("y_sycl", y_sycl[0].to("cpu"))
