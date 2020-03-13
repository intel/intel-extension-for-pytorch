import torch

import torch_ipex

x_cpu = torch.tensor([[1,2,3,4]], device=torch.device("cpu"), dtype=torch.float)
x_sycl = torch.tensor([[1,2,3,4]], device=torch.device("dpcpp"), dtype=torch.float)

y_cpu = x_cpu.masked_fill(mask = torch.BoolTensor([True,True,False,False]), value=torch.tensor(-1e9))
y_sycl = x_sycl.masked_fill(mask = torch.BoolTensor([True,True,False,False]).to("dpcpp"), value=torch.tensor(-1e9))

print("y_cpu = ", y_cpu)
print("y_sycl = ", y_sycl.cpu())

