import numpy
import torch
import torch_ipex

dtype_float = torch.float
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

user_cpu = torch.randn([256, 3, 2, 4], device=cpu_device, dtype=dtype_float)
print(user_cpu)
res_cpu = torch.sum(user_cpu, 0, False)
print("cpu result:")
print(res_cpu)
print("begin sycl compute:")
res_sycl = torch.sum(user_cpu.to("dpcpp"), 0, False)
print("sycl result:")
print(res_sycl.cpu())
