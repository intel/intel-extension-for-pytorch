import torch
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

x_cpu = torch.randn(4, device=cpu_device)
y_cpu = torch.randn(4, device=cpu_device)

dist = torch.atan2(x_cpu, y_cpu)

print("torch.atan2(x_cpu, y_cpu)", dist)

x_dpcpp = x_cpu.to(dpcpp_device)
y_dpcpp = y_cpu.to(dpcpp_device)

dist = torch.atan2(x_dpcpp, y_dpcpp)

print("torch.atan2(x_dpcpp, y_dpcpp)", dist.cpu())
