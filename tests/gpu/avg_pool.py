import numpy
import torch
import torch.nn as nn
import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x_cpu = torch.ones([2, 2, 3, 3], device=cpu_device, dtype=dtype)
grad_cpu = torch.ones([2, 2, 3, 3], device=cpu_device, dtype=dtype)

avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=True)

y_cpu = conv1(x_cpu)
y_cpu = avg_pool(y_cpu)
print("y_cpu", y_cpu)

conv1.to("dpcpp")
x_sycl = x_cpu.to("dpcpp")
y_sycl = conv1(x_sycl)
y_sycl = avg_pool(y_sycl)
print("y_sycl", y_sycl.to("cpu"))
