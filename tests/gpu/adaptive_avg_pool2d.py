import numpy
import torch
import torch.nn as nn

import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x_cpu = torch.ones([1, 1, 8, 8], device=cpu_device, dtype=dtype)
grad_cpu = torch.ones([1, 1, 2, 2], device=cpu_device, dtype=dtype)
x_sycl = x_cpu.to("dpcpp")
grad_sycl = grad_cpu.to("dpcpp")

avg_pool = nn.AdaptiveAvgPool2d((2,2))
#conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)



x_cpu.requires_grad = True

#y_cpu = conv1(x_cpu)
y_cpu = avg_pool(x_cpu)
print("y_cpu", y_cpu)
#conv1.zero_grad()
#output_cpu = y_cpu.backward(grad_cpu)
#print("x_cpu.grad", x_cpu.grad)

x_sycl.requires_grad = True
#conv1 = conv1.sycl()
#y_sycl = conv1(x_sycl)
y_sycl = avg_pool(x_sycl)
print("y_sycl", y_sycl.cpu())
#conv1.zero_grad()
#output_sycl = y_sycl.backward(grad_sycl)
#print("x_sycl.grad", x_sycl.grad.cpu())
