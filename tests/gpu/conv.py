import numpy
import torch
import torch.nn as nn

import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

# functionality
x_cpu = torch.ones([8, 8, 24, 24], device=sycl_device, dtype=dtype)
grad_cpu = torch.ones([8, 8, 24, 24], device=sycl_device, dtype=dtype)
conv1 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)

conv1.sycl()
conv2.sycl()
y_sycl = conv1(x_sycl)
y_sycl = conv2(y_sycl)
