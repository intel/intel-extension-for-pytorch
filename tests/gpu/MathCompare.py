import numpy
import torch
import torch.nn as nn

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x=torch.tensor([[True, True], [True, True]], device=cpu_device)
y=torch.tensor([[1, 2], [3, 4]], device=cpu_device)
z=torch.tensor([[1, 1], [4, 4]], device=cpu_device)
x=torch.lt(y, z)
x2=torch.eq(y, z)
x3=torch.gt(y, z)
x4=torch.le(y, z)
x5=torch.ne(y, z)
x6=torch.ge(y, z)

print("cpu: ")
print(x)
print(x2)
print(x3)
print(x4)
print(x5)
print(x6)
x_sycl=x.to("dpcpp")
y_sycl=y.to("dpcpp")
z_sycl=z.to("dpcpp")
x_out=torch.lt(y_sycl, z_sycl)
x2_out=torch.eq(y_sycl, z_sycl)
x3_out=torch.gt(y_sycl, z_sycl)
x4_out=torch.le(y_sycl, z_sycl)
x5_out=torch.ne(y_sycl, z_sycl)
x6_out=torch.ge(y_sycl, z_sycl)

print("sycl: ")
print(x_out.to("cpu"))
print(x2_out.to("cpu"))
print(x3_out.to("cpu"))
print(x4_out.to("cpu"))
print(x5_out.to("cpu"))
print(x6_out.to("cpu"))
