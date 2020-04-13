import torch
import torch.nn as nn

import torch_ipex

device = torch.device("dpcpp")
dtype = torch.float

m1_cpu = torch.randn([3, 4], dtype=dtype)
m2_cpu = torch.randn([4, 2], dtype=dtype)
x_cpu = torch.ones([3, 2], dtype=dtype)

m1_dpcpp = m1_cpu.to(device)
m2_dpcpp = m2_cpu.to(device)
x_dpcpp = x_cpu.to(device)

print("cpu addmm_ self", x_cpu)
x_cpu.addmm_(m1_cpu, m2_cpu)
print("cpu addmm_ result", x_cpu)

print("dpcpp addmm_ self", x_dpcpp.cpu())
x_dpcpp.addmm_(m1_dpcpp, m2_dpcpp)
print("dpcpp addmm_ result", x_dpcpp.cpu())

