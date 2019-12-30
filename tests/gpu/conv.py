import numpy
import torch
import torch.nn as nn

dtype = torch.float
cpu_device = torch.device("cpu")
sycl_device = torch.device("sycl")

# functionality
x_cpu = torch.ones([8, 8, 24, 24], device=cpu_device, dtype=dtype)
grad_cpu = torch.ones([8, 8, 24, 24], device=cpu_device, dtype=dtype)
conv1 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)

x_sycl = x_cpu.to("sycl")
conv1.sycl()
conv2.sycl()
y_sycl = conv1(x_sycl)
y_sycl = conv2(y_sycl)
#print (y_sycl.shape)
#print("y_sycl=", y_sycl)

conv1.zero_grad()
grad_sycl = grad_cpu.to("sycl")
y_sycl.backward(grad_sycl)

print(y_sycl.device)

# forward validation
x_cpu = torch.ones([2, 2, 3, 3], device=cpu_device, dtype=dtype)
grad_cpu = torch.ones([2, 2, 3, 3], device=cpu_device, dtype=dtype)
conv = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)

y_cpu = conv(x_cpu)
print(y_cpu)
conv.sycl()
y_sycl = conv(x_cpu.sycl())
print(y_sycl.cpu())

# backward validation
# ...
