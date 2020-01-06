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

ref_y = conv1(ref_x)

conv1.to("dpcpp")
y = conv1(x)

print("ref: ")
print(ref_y)
print("real: ")
print(y.to("cpu"))
