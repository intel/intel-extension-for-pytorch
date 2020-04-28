import torch
import torch.nn as nn

import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

x_cpu = torch.randn([2, 2, 4, 4], device=cpu_device, dtype=dtype)
x_dpcpp = x_cpu.to("dpcpp")
grad_cpu = torch.randn([2, 2, 2, 2], device=cpu_device)
grad_dpcpp = grad_cpu.to("dpcpp")
max_pool = nn.FractionalMaxPool2d(2, output_size=(2, 2), return_indices=True)

#cpu
x_cpu.requires_grad_(True)
y_cpu = max_pool(x_cpu)
print("y_cpu", y_cpu[0])
y_cpu[0].backward(grad_cpu)
print("y_cpu backward", x_cpu.grad)

max_pool = nn.FractionalMaxPool2d(2, output_size=(2, 2), return_indices=True)
max_pool.to("dpcpp")
x_dpcpp.requires_grad_(True)
y_dpcpp = max_pool(x_dpcpp)

print("y_dpcpp", y_dpcpp[0].cpu())
grad_dpcpp = grad_cpu.to("dpcpp")
y_dpcpp[0].backward(grad_dpcpp)
print("y_dpcpp backward", x_dpcpp.grad.cpu())

