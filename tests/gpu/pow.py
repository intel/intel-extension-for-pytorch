import torch
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

x_cpu = torch.tensor(([2.5, 3.1, 1.3]), dtype=torch.float, device = cpu_device)
x_dpcpp = torch.tensor(([2.5, 3.1, 1.3]), dtype=torch.float, device = dpcpp_device)

y_cpu = torch.tensor(([3.0, 3.0, 3.0]), dtype=torch.float, device = cpu_device)
y_dpcpp = torch.tensor(([3.0, 3.0, 3.0]), dtype=torch.float, device = dpcpp_device)

print("pow x y cpu", torch.pow(x_cpu, y_cpu))
print("pow x y sycl", torch.pow(x_dpcpp, y_dpcpp).cpu())

print("x.pow y cpu", x_cpu.pow(y_cpu))
print("x.pow y sycl", x_dpcpp.pow(y_dpcpp).cpu())

print("x.pow_ y cpu", x_cpu.pow_(y_cpu))
print("x.pow_ y sycl", x_dpcpp.pow_(y_dpcpp).cpu())
