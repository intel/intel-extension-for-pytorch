import numpy
import torch
import torch.nn as nn

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")
a = torch.randn(4)
b=a.to("dpcpp")
print(a)
print("cpu")
print(torch.sigmoid(a))
print("sycl")
print(torch.sigmoid(b).to("cpu"))
