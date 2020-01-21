import numpy
import torch

import torch_ipex

from torch import nn


cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

# functionality
x_ref = torch.ones([2, 2], device=cpu_device)
x_ref[0][0] = 1
x_ref[0][1] = 3
x_ref[1][0] = 2
x_ref[1][1] = 1

y_ref = nn.Threshold(2, 0)(x_ref)
print(y_ref)
x = x_ref.to("dpcpp")
y = nn.Threshold(2, 0)(x)
print(y.to("cpu"))
