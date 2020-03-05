import numpy
import torch
import torch.nn as nn

import torch_ipex

dtype_float = torch.float
cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

user_cpu = torch.tensor([[1.11, 2.22, 3.33], [4.44, 5.55, 6.66]], device=cpu_device, dtype=dtype_float, requires_grad=True)
m = nn.Sigmoid()
cpu_res = m(user_cpu)
print(cpu_res)
cpu_res.backward(torch.tensor([[0.5, 1.0, 1.5],[2.0, 2.5, 3.0]], device=cpu_device, dtype=dtype_float))
print(user_cpu.grad)

user_dpcpp = torch.tensor([[1.11, 2.22, 3.33], [4.44, 5.55, 6.66]], device=dpcpp_device, dtype=dtype_float, requires_grad=True)
dpcpp_res = m(user_dpcpp)
print(dpcpp_res.to("cpu"))
dpcpp_res.backward(torch.tensor([[0.5, 1.0, 1.5],[2.0, 2.5, 3.0]], device=dpcpp_device, dtype=dtype_float))
print(user_dpcpp.grad.to("cpu"))
