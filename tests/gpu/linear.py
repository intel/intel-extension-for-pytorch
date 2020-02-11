import numpy
import torch
import torch.nn as nn

import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

# functionality
x_cpu = torch.ones([3, 4], device=cpu_device, dtype=dtype)
grad_cpu = torch.ones([3, 2], device=cpu_device, dtype=dtype)
linear1 = nn.Linear(4, 2, bias=True)

print(x_cpu)
y_cpu = linear1(x_cpu)
print(y_cpu)

print("--------------------------------------------------------------------")

# linear1.zero_grad()
linear1.to("dpcpp")

x_sycl = x_cpu.to("dpcpp")
print(x_sycl.cpu())
y_sycl = linear1(x_sycl)
print(y_sycl.cpu())
