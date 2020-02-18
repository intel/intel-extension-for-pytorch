import numpy
import torch
import torch.nn as nn

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

#x=torch.tensor([1,1,1,1,1], device=cpu_device)
x=torch.logspace(start=-10, end=10, steps=5, device=cpu_device)
y=torch.linspace(start=-10, end=10, steps=5, device=cpu_device)
z=torch.arange(1, 2.5, 0.5, device=cpu_device)
n=torch.range(1, 2.5, 0.5, device=cpu_device)

#x_sycl=x.to("dpcpp")
x_out=torch.logspace(start=-10, end=10, steps=5, device=sycl_device)
y_out=torch.linspace(start=-10, end=10, steps=5, device=sycl_device)
z_out=torch.arange(1, 2.5, 0.5, device=sycl_device)
n_out=torch.range(1, 2.5, 0.5, device=sycl_device)

print("cpu: ")
print(x)
print(y)
print(z)
print(n)

print("sycl: ")
print(x_out.to("cpu"))
print(y_out.to("cpu"))
print(z_out.to("cpu"))
print(n_out.to("cpu"))