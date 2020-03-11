import torch

import torch_ipex
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x1 = torch.rand(2, 10, device = cpu_device)
x2 = torch.ones(3, 10, device = cpu_device)
x1_sycl = x1.to("dpcpp")
x2_sycl = x2.to("dpcpp")
x2.scatter_(0, torch.tensor([[0, 1, 2, 0, 0, 0, 1, 2, 0, 0], [2, 0, 0, 1, 2, 2, 0, 0, 1, 2]], device = cpu_device), x1)
print(x2)

x2_sycl.scatter_(0, torch.tensor([[0, 1, 2, 0, 0, 0, 1, 2, 0, 0], [2, 0, 0, 1, 2, 2, 0, 0, 1, 2]], device = sycl_device), x1_sycl)
print(x2_sycl.cpu())

x1 = torch.rand(2, 10, device = cpu_device)
x2 = torch.ones(3, 10, device = cpu_device)
x1_sycl = x1.to("dpcpp")
x2_sycl = x2.to("dpcpp")
x2.scatter_add_(0, torch.tensor([[0, 1, 2, 0, 0, 0, 1, 2, 0, 0], [2, 0, 0, 1, 2, 2, 0, 0, 1, 2]], device = cpu_device), x1)
print(x2)

x2_sycl.scatter_add_(0, torch.tensor([[0, 1, 2, 0, 0, 0, 1, 2, 0, 0], [2, 0, 0, 1, 2, 2, 0, 0, 1, 2]], device = sycl_device), x1_sycl)
print(x2_sycl.cpu())


x1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device = cpu_device)
x2 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32, device = cpu_device)
x1_sycl = x1.to("dpcpp")
x2_sycl = x2.to("dpcpp")
x2.scatter_add_(0, torch.tensor([[0, 1], [2, 0]], device = cpu_device), x1)
print(x2)

x2_sycl.scatter_add_(0, torch.tensor([[0, 1], [2, 0]], device = sycl_device), x1_sycl)
print(x2_sycl.cpu())

