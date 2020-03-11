import torch
import time
import torch_ipex

x_cpu = torch.randn(3, 4)
sorted_cpu, indices_cpu = torch.sort(x_cpu)
print("x_cpu = ", x_cpu, "sorted = ", sorted_cpu, "indices = ", indices_cpu)

x_dpcpp = x_cpu.to("dpcpp")
sorted_dpcpp, indices_dpcpp = torch.sort(x_dpcpp)
print("x_dpcpp = ", x_dpcpp.cpu(), "sorted_dpcpp = ", sorted_dpcpp.cpu(), "indices_dpcpp", indices_dpcpp.cpu())
