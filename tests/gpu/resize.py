import numpy
import torch
import torch.nn as nn

import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x = torch.ones([2,2,4,3], device=sycl_device, dtype=dtype)
x.resize_(1,2,3,4)

y = torch.ones([2,2,4,3], device=cpu_device, dtype=dtype)
y.resize_(1,2,3,4)

print("sycl: ")
print(x.to("cpu"))
print("cpu: ")
print(y)