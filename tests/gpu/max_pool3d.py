import numpy
import torch
import torch.nn as nn
import torch_ipex

dtype = torch.float
dtype1 = torch.long
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x_cpu = torch.randn([1, 4, 4, 3], device=cpu_device, dtype=dtype)
grad_cpu = torch.randn([1, 4, 4, 3], device=cpu_device, dtype=dtype)
x_sycl = x_cpu.to("dpcpp")
grad_sycl = grad_cpu.to("dpcpp")

max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, return_indices=True)

x_cpu.requires_grad_(True)
y_cpu = max_pool(x_cpu)
print("y_cpu", y_cpu[0])
output_cpu = y_cpu[0].backward(grad_cpu)
print("x_cpu.grad", x_cpu.grad)


max_pool.to("dpcpp")
x_sycl.requires_grad_(True)
y_sycl = max_pool(x_sycl)
print("y_sycl", y_sycl[0].to("cpu"))
output_sycl = y_sycl[0].backward(grad_sycl)
print("x_sycl.grad", x_sycl.grad.to("cpu"))
