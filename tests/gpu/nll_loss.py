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
x = torch.tensor((0.5), dtype=torch.float)
input_dpcpp = input.to("dpcpp")
target_dpcpp = target.to("dpcpp")
x_dpcpp = x.to("dpcpp")
input.requires_grad = True
output = F.nll_loss(input, target)
output.backward(x)
print("CPU: ", output)
print("CPU: ", input.grad)

input_dpcpp.requires_grad = True
output_dpcpp = F.nll_loss(input_dpcpp, target_dpcpp)
output_dpcpp.backward(x_dpcpp)
print("SYCL: ", output.to("cpu"))
print("SYCL: ", input_dpcpp.grad.to("cpu"))
