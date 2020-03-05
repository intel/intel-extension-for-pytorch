import torch
import torch.nn as nn
import torch_ipex


dtype = torch.float
cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


#cpu
linear = nn.Linear(4, 2)
tanh = nn.Tanh()

print("linear weight", linear.weight)

x_cpu = torch.tensor([[1.23, 2.34, 6.45, 2.22], [0.23, 1.34, 7.45, 1.22]], device=cpu_device, dtype=dtype)
print("x_cpu", x_cpu)

z_cpu = linear(x_cpu)
print("z_cpu", z_cpu)

y_cpu = tanh(z_cpu)
print("y_cpu", y_cpu)

y_cpu.backward(torch.tensor([[1.01, 8.32], [2.4, 3.22]], device=cpu_device))
print("linear grad", linear.weight.grad)

#dpcpp
linear_dpcpp = linear.to("dpcpp")
tanh_dpcpp = tanh.to("dpcpp")

print("dpcpp linear weight", linear_dpcpp.weight.to("cpu"))

x_dpcpp = x_cpu.to("dpcpp")
print("x_dpcpp", x_dpcpp.to("cpu"))

z_dpcpp = linear_dpcpp(x_dpcpp)
print("z_dpcpp", z_dpcpp.to("cpu"))

y_dpcpp = tanh(z_dpcpp)
print("y_dpcpp", y_dpcpp.to("cpu"))

y_dpcpp.backward(torch.tensor([[1.01, 8.32], [2.4, 3.22]], device=dpcpp_device))
print("dpcpp linear grad", linear.weight.grad.to("cpu"))
