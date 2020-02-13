from __future__ import print_function
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x_i = torch.randn([2,2,3,3], device=cpu_device, dtype=dtype)
grad_i = torch.randn([2,2,3,3], device=cpu_device, dtype=dtype)

x_sycl_i = x_i.to("dpcpp")
grad_sycl_i = grad_i.to("dpcpp")

x_cpu = Variable(x_i, requires_grad = True)
grad_cpu = Variable(grad_i, requires_grad = True)
bn1 = nn.BatchNorm2d(2)
bn2 = nn.BatchNorm2d(2)
y_cpu1 = bn1(x_cpu)
y_cpu = bn2(y_cpu1)


# y_cpu.backward(grad_cpu)


print("x_cpu = ", y_cpu)
# print("x_cpu.grad = ", x_cpu.grad)


x_sycl = Variable(x_sycl_i, requires_grad = True)
grad_sycl = Variable(grad_sycl_i, requires_grad = True)
bn1.to("dpcpp")
bn2.to("dpcpp")

y_sycl1 = bn1(x_sycl)
y_sycl =  bn2(y_sycl1)

# y_sycl.backward(grad_sycl)

# y = y_sycl1.cpu()
# y = Variable(y, requires_grad = True)
# y.backward(grad_cpu)
print("y_sycl = ", y_sycl.cpu())
# print("x_sycl.grad", x_sycl.grad.cpu())
