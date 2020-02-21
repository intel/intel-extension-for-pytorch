import numpy
import torch
import torch.nn as nn

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

a = torch.arange(9, dtype= torch.float) - 4
b = a.reshape((3, 3))
a_sycl=a.to("dpcpp")
b_sycl=b.to("dpcpp")
print("CPU")
print(torch.norm(a))

print(torch.norm(b))

print("SYCL")
print(torch.norm(a_sycl).to("cpu"))

print(torch.norm(b_sycl).to("cpu"))
