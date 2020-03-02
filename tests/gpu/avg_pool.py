import numpy
import torch
import torch.nn as nn
import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x_cpu = torch.ones([2, 2, 3, 3], device=cpu_device, dtype=dtype)
grad_cpu = torch.ones([2, 2, 3, 3], device=cpu_device, dtype=dtype)
x_sycl = x_cpu.to("dpcpp")
grad_sycl = grad_cpu.to("dpcpp")

avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=True)

x_cpu.requires_grad_(True)
# y_cpu = conv1(x_cpu)
y_cpu = avg_pool(x_cpu)
print("y_cpu", y_cpu)
# conv1.zero_grad()
output_cpu = y_cpu.backward(grad_cpu)
print("x_cpu.grad", x_cpu.grad)

# conv1.to("dpcpp")
avg_pool.to("dpcpp")
x_sycl.requires_grad_(True)
# y_sycl = conv1(x_sycl)
y_sycl = avg_pool(x_sycl)
print("y_sycl", y_sycl.to("cpu"))
# conv1.zero_grad()
output_sycl = y_sycl.backward(grad_sycl)
print("x_sycl.grad", x_sycl.grad.to("cpu"))
