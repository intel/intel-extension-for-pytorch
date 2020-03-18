import numpy
import torch
import torch.nn as nn

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")


x = torch.tensor([[ 0.6580, -1.0969, -0.4614], [-0.1034, -0.5790,  0.1497]], device=cpu_device)
#x_sycl = torch.tensor([[ 0.6580, -1.0969, -0.4614], [-0.1034, -0.5790,  0.1497]], device = sycl_device)
x_sycl = x.to("dpcpp")

y =  torch.addcdiv(x, 0.1, x, x)

print("addcdiv cpu:", y)
y_sycl = torch.addcdiv(x_sycl, 0.1, x_sycl, x_sycl)
print("addcdiv sycl:", y_sycl.cpu())

y = torch.addcmul(x, 0.1, x, x)

y_sycl = torch.addcmul(x_sycl, 0.1, x_sycl, x_sycl)
print("addcmul cpu:", y)
print("addcdiv sycl: ", y_sycl.cpu())

y = torch.lerp(x,x, 0.5)
y_sycl = torch.lerp(x_sycl,x_sycl, 0.5)
print("lerp cpu:", y)
print("lerp sycl: ", y_sycl.cpu())
