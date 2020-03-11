import numpy
import torch
import torch_ipex

dtype_float = torch.float
cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

user_cpu1 = torch.randn([2, 2, 3], device=cpu_device, dtype=dtype_float)
user_cpu2 = torch.randn([2, 2, 3], device=cpu_device, dtype=dtype_float)
user_cpu3 = torch.randn([2, 2, 3], device=cpu_device, dtype=dtype_float)

res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=1);
print("CPU Result:")
print(res_cpu)

res_dpcpp = torch.cat((user_cpu1.to("dpcpp"), user_cpu2.to("dpcpp"), user_cpu3.to("dpcpp")), dim=1);
print("SYCL Result:")
print(res_dpcpp.cpu())
