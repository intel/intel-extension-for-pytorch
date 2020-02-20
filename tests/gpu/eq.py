import torch
import time
import torch_ipex

x_cpu1 = torch.tensor([[1., 2.], [3., 4.]])
x_cpu2 = torch.tensor([[1., 1.], [4., 4.]])
y_cpu = torch.eq(x_cpu1, x_cpu2)
print(y_cpu)

x_sycl1 = x_cpu1.to("dpcpp")
x_sycl2 = x_cpu2.to("dpcpp")

y_sycl = torch.eq(x_sycl1, x_sycl2)
print(y_sycl.cpu())
