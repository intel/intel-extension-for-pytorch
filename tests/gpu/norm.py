import numpy
import torch
import torch.nn as nn

import torch_ipex

device = torch.device("dpcpp")

x_cpu = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=torch.float32)
print(x_cpu.norm(p='fro', dim=[0, 1]))
print(x_cpu.norm(p='fro', dim=[0]))

x_dpcpp = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=torch.float32, device=device)
print(x_dpcpp.norm(p='fro', dim=[0, 1]).cpu())
print(x_dpcpp.norm(p='fro', dim=[0]).cpu())
