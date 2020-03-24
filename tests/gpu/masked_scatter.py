import torch

import torch_ipex

dtype_float = torch.float
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x_cpu = torch.rand([2,3,4], device=cpu_device, dtype=dtype_float)
y_cpu = torch.rand([2,3,4], device=cpu_device, dtype=dtype_float)
mask_cpu = y_cpu.ge(0.5)
z_cpu = torch.zeros_like(x_cpu)

z_cpu.masked_scatter_(mask_cpu, x_cpu)
print("z_cpu:")
print(z_cpu)

z_sycl = z_cpu.to("dpcpp")
z_sycl.masked_scatter_(mask_cpu.to("dpcpp"), x_cpu.to("dpcpp"))
print("z_sycl:")
print(z_sycl.to("cpu"))

### For debug
#print("mask_cpu:")
#print(mask_cpu)

#print("y_cpu:")
#print(y_cpu)


