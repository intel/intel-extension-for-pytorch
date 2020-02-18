from __future__ import print_function
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_ipex

# input is of size N x C = 3 x 5
input = torch.randn(3, 5)
# each element in target has to have 0 <= value < C
target = torch.tensor([1, 0, 4])
output = F.nll_loss(input, target)
print("CPU: ", output)

output = F.nll_loss(input.to("dpcpp"), target.to("dpcpp"))
print("SYCL: ", output.to("cpu"))
