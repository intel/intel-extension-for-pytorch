import numpy
import torch
import torch_ipex

dtype_float = torch.float
cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

user_cpu = torch.randn([2, 2, 2, 2, 2], device=cpu_device, dtype=dtype_float)
print(user_cpu)
res_cpu = torch.mean(user_cpu, (0,4), False)
print("cpu result:")
print(res_cpu)
print("begin dpcpp compute:")
res_dpcpp = torch.mean(user_cpu.to("dpcpp"), (0,4), False)
print("dpcpp result:")
print(res_dpcpp.cpu())
