import numpy
import torch
import torch.nn as nn

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x_cpu = torch.tensor([[[[1,2,3,4,5],[4,5,6,7,8]],[[1,2,3,4,5],[4,5,6,7,8]]],[[[1,2,3,4,5],[4,5,6,7,8]],[[1,2,3,4,5],[4,5,6,7,8]]]], dtype=torch.float32, device = cpu_device)
#x_sycl = torch.tensor([[[[1,2,3,4,5],[4,5,6,7,8]],[[1,2,3,4,5],[4,5,6,7,8]]],[[[1,2,3,4,5],[4,5,6,7,8]],[[1,2,3,4,5],[4,5,6,7,8]]]], dtype=torch.float32, device = sycl_device)
x_sycl=x_cpu.to("dpcpp")

print("cpu result", torch.nn.functional.upsample_nearest(x_cpu,[2,5]))
print("sycl result", torch.nn.functional.upsample_nearest(x_sycl,[2,5]).cpu())

print("cpu result", torch.nn.functional.upsample_nearest(x_cpu,[4,10]))
print("sycl result", torch.nn.functional.upsample_nearest(x_sycl,[4,10]).cpu())

print("cpu result", torch.nn.functional.upsample_nearest(x_cpu,[3,8]))
print("sycl result", torch.nn.functional.upsample_nearest(x_sycl,[3,8]).cpu())

print("cpu result", torch.nn.functional.upsample_nearest(x_cpu,[1,3]))
print("sycl result", torch.nn.functional.upsample_nearest(x_sycl,[1,3]).cpu())