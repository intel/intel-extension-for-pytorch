import numpy
import torch
import torch.nn as nn

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x=torch.empty_strided((2, 3), (1, 2), device=cpu_device)
x_out=torch.empty_strided((2, 3), (1, 2), device=sycl_device)
y=torch.eye(3, device=cpu_device)
y_out=torch.eye(3, device=sycl_device)
m = torch.tril_indices(3, 3, device=cpu_device)
n = torch.triu_indices(3, 3, device=cpu_device)
m_out = torch.tril_indices(3, 3, device=sycl_device)
n_out = torch.triu_indices(3, 3, device=sycl_device)

print("cpu: ")
print(x)
print(x.stride())
print(x.size())
print(y)
print(m)
print(n)

print("sycl: ")
print(x_out.to("cpu"))
print(x_out.stride())
print(x_out.size())
print(y_out.to("cpu"))
print(m_out.to("cpu"))
print(n_out.to("cpu"))
