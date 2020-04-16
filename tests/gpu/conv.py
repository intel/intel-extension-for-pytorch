import numpy
import torch
import torch.nn as nn

import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

# functionality
x = torch.ones([1, 2, 3, 3], device=sycl_device, dtype=dtype)
# grad = torch.ones([8, 8, 24, 24], device=sycl_device, dtype=dtype)
conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
# conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)

ref_x = torch.ones([1, 2, 3, 3], device=cpu_device, dtype=dtype)
# ref_grad = torch.ones([8, 8, 24, 24], device=sycl_device, dtype=dtype)

grad_cpu = torch.ones([1, 2, 3, 3], device=cpu_device, dtype=dtype)
grad_sycl = x = torch.ones([1, 2, 3, 3], device=sycl_device, dtype=dtype)

ref_x.requires_grad_(True)
ref_y = conv1(ref_x)
conv1.zero_grad()
output_cpu = ref_y.backward(grad_cpu)
print("ref: ")
print(ref_y)
print("ref grad: ")
print(ref_x.grad[0])

x.requires_grad_(True)
conv1.to("dpcpp")
y = conv1(x)
conv1.zero_grad()
output_sycl = y.backward(grad_sycl)
print("real: ")
print(y.to("cpu"))
print("real grad: ")
print(x.grad[0].to("cpu"))


x = torch.randn([1, 2, 2, 1, 1], device=cpu_device, dtype=dtype, requires_grad=True)
grad = torch.ones([1, 2, 2, 1, 1], device=cpu_device, dtype=dtype, requires_grad=True)
conv3 = nn.Conv3d(2, 2, kernel_size=3, stride=1, padding=1, bias=True)
y = conv3(x)
y.backward(grad)

conv3.to(sycl_device)
x_dpcpp = x.to(sycl_device)
y_dpcpp = conv3(x_dpcpp)
grad_dpcpp = grad.to(sycl_device)

y_dpcpp.backward(grad_dpcpp)

print("ref: ")
print(y)
print("ref backward: ")
print(x)

print("real: ")
print(y_dpcpp.to(cpu_device))
print("real backward: ")
print(x_dpcpp.to(cpu_device))