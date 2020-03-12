import torch

import torch_ipex

x_cpu1 = torch.randn(3, device=torch.device("cpu"), dtype=torch.float)
y1_cpu1 = torch.diag(x_cpu1)
y2_cpu1 = torch.diag(x_cpu1, 1)
print("cpu1 y1", y1_cpu1)
print("cpu1 y2", y2_cpu1)

x_sycl1 = x_cpu1.to("dpcpp")
y1_sycl1 = torch.diag(x_sycl1)
y2_sycl1 = torch.diag(x_sycl1, 1)
print("syc1 y1", y1_sycl1.cpu())
print("syc1 y2", y2_sycl1.cpu())

x_cpu2 = torch.randn(3, 3, device=torch.device("cpu"), dtype=torch.float)
y1_cpu2 = torch.diag(x_cpu2)
y2_cpu2 = torch.diag(x_cpu2, 1)
print("cpu2 y1", y1_cpu2)
print("cpu2 y2", y2_cpu2)

x_sycl2 = x_cpu2.to("dpcpp")
y1_sycl2 = x_sycl2.diag(0)
y2_sycl2 = x_sycl2.diag(1)
print("syc2 y1", y1_sycl2.cpu())
print("syc2 y2", y2_sycl2.cpu())

