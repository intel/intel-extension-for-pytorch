import torch
import torch.nn as nn

import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

x_cpu = torch.randn([1, 4, 4, 4], device=cpu_device, dtype=dtype)
grad_cpu = torch.ones([1, 4, 4, 4], device=cpu_device, dtype=dtype)
x_dpcpp = x_cpu.to("dpcpp")
grad_cpu = torch.randn([1, 2, 2, 2], device=cpu_device)
grad_dpcpp = grad_cpu.to("dpcpp")
avg_pool = nn.AdaptiveAvgPool3d((2, 2, 2))

#cpu
x_cpu.requires_grad_(True)
y_cpu = avg_pool(x_cpu)
print("y_cpu", y_cpu)
y_cpu.backward(grad_cpu)
print("y_cpu backward", x_cpu.grad)


avg_pool.to("dpcpp")
x_dpcpp.requires_grad_(True)
y_dpcpp = avg_pool(x_dpcpp)

print("y_dpcpp", y_dpcpp.cpu())

grad_dpcpp = grad_cpu.to("dpcpp")
y_dpcpp.backward(grad_dpcpp)
print("y_dpcpp backward", x_dpcpp.grad.cpu())