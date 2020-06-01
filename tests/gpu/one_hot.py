import torch
import torch.nn.functional as F

import torch_ipex

x_cpu = torch.arange(0, 5) %3
x_dpcpp = x_cpu.to("dpcpp")
y_cpu = F.one_hot(x_cpu, num_classes=5)
print(y_cpu)
y_dpcpp = F.one_hot(x_dpcpp, num_classes=5)
print(y_dpcpp.to("cpu"))
