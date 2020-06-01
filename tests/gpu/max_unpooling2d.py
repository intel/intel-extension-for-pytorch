import torch
import torch.nn as nn

import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

input = torch.randn([2, 2, 4, 4], device=cpu_device, dtype=dtype)
pool = nn.MaxPool2d(2, stride=2, return_indices=True)
output, indices = pool(input)

x_cpu = output
x_dpcpp = output.to("dpcpp")
indices_dpcpp = indices.to("dpcpp")
grad_cpu = torch.randn([2, 2, 4, 4], device=cpu_device)
grad_dpcpp = grad_cpu.to("dpcpp")
output_size=torch.Size([2, 2, 5, 5])
unpool = nn.MaxUnpool2d(2, stride=2)

x_cpu.requires_grad_(True)
y_cpu = unpool(x_cpu, indices)
print("y_cpu", y_cpu)
y_cpu.backward(grad_cpu)
print("y_cpu backward", x_cpu.grad)


unpool.to("dpcpp")
x_dpcpp.requires_grad_(True)
y_dpcpp = unpool(x_dpcpp, indices_dpcpp)
print("y_dpcpp", y_dpcpp.to("cpu"))
y_dpcpp.backward(grad_dpcpp)
print("y_dpcpp backward", x_dpcpp.grad.to("cpu"))


