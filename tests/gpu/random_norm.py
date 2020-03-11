import torch
import torch.nn as nn
import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

x_cpu = torch.tensor([1.111, 2.222, 3.333, 4.444, 5.555, 6.666], device=cpu_device, dtype=dtype);
x_dpcpp = torch.tensor([1.111, 2.222, 3.333, 4.444, 5.555, 6.666], device=dpcpp_device, dtype=dtype);

print("normal_ cpu", x_cpu.normal_(2.0, 0.5))
print("normal_ dpcpp", x_dpcpp.normal_(2.0, 0.5).cpu())
