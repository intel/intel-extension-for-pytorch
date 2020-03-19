from __future__ import print_function
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


layer_norm = nn.LayerNorm([1, 3, 3])
x_i = torch.randn([1,1,3,3], device=cpu_device, dtype=dtype)
grad_i = torch.randn([1,1,3,3], device=cpu_device, dtype=dtype)

x_i[0][0][0][0] = 0.5021
x_i[0][0][0][1] = -0.9922
x_i[0][0][0][2] = -0.7365
x_i[0][0][1][0] = 0.0629
x_i[0][0][1][1] = -2.0536
x_i[0][0][1][2] = -0.9989
x_i[0][0][2][0] = 0.4911
x_i[0][0][2][1] = 0.9744
x_i[0][0][2][2] = -1.9760

grad_i[0][0][0][0] = 0.6259
grad_i[0][0][0][1] = -0.3097
grad_i[0][0][0][2] = -0.8985
grad_i[0][0][1][0] = 0.0328
grad_i[0][0][1][1] = 1.9637
grad_i[0][0][1][2] = -1.7078
grad_i[0][0][2][0] = 0.3252
grad_i[0][0][2][1] = -0.2873
grad_i[0][0][2][2] = -0.4864

# torch.save(layer_norm, "./log/layer_norm.pt")
# torch.save(x_i, "./log/layer_norm_x.pt")
# torch.save(grad_i, "./log/layer_norm_grad.pt")

x_dpcpp_i = x_i.to("dpcpp")
grad_dpcpp_i = grad_i.to("dpcpp")

x_cpu = Variable(x_i, requires_grad = True)
y_cpu = layer_norm(x_cpu)

y_cpu.backward(grad_i)

print("x_cpu = ", x_cpu)
print("layer_norm = ", layer_norm.weight.cpu())
print("y_cpu = ", y_cpu)
print("x_cpu.grad = ", x_cpu.grad)
print("layer_norm.grad = ", layer_norm.weight.grad)
x_cpu.grad.detach()
x_cpu.grad.zero_()

# layer_norm_dpcpp = torch.load("./log/layer_norm.pt").to(dpcpp_device)
layer_norm_dpcpp = layer_norm.to(dpcpp_device)
layer_norm.zero_grad()

x_dpcpp = Variable(x_dpcpp_i, requires_grad = True)
y_dpcpp = layer_norm_dpcpp(x_dpcpp)

y_dpcpp.backward(grad_dpcpp_i)

print("x_dpcpp = ", x_dpcpp.cpu())
print("layer_norm_dpcpp = ", layer_norm_dpcpp.weight.cpu())
print("y_dpcpp = ", y_dpcpp.cpu())
print("x_dpcpp.grad = ", x_dpcpp.grad.cpu())
print("layer_norm_dpcpp.grad = ", layer_norm_dpcpp.weight.grad.cpu())
