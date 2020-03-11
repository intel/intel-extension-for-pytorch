import torch

import torch_ipex

cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")

x = torch.ones([5,3], device = cpu_device)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
x.index_fill_(0, index, -2)
print("x = ", x)

x_sycl = torch.ones([5,3], device = sycl_device)
index = torch.tensor([0, 4, 2], device = sycl_device)
x_sycl.index_fill_(0, index, -2)

print("x_sycl = ", x_sycl.to("cpu"))