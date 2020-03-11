import torch

import torch_ipex

#test = torch.tensor(3, device = sycl_device)
#test.item()
cpu_device = torch.device("cpu")
sycl_device = torch.device("dpcpp")
test = torch.tensor(3, device = sycl_device)
test.item()


x = torch.ones([5,3], device = cpu_device)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
x.index_add_(0, index, t)
print("x = ", x)

x_sycl = torch.ones([5,3], device = sycl_device)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float, device = sycl_device)
index = torch.tensor([0, 4, 2], device = sycl_device)
x_sycl.index_add_(0, index, t)
print("x_sycl = ", x_sycl.to("cpu"))

# x_sycl = torch.ones([3,5], device = sycl_device)
# t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float, device = sycl_device)
# index = torch.tensor([0, 4, 2], device = sycl_device)
# x_sycl.index_add_(1, index, t)
# print("x_sycl = ", x_sycl.to("cpu"))

# x_sycl = torch.ones([5,1], device = sycl_device)
# t = torch.tensor([[100], [100], [100], [100], [100]], dtype=torch.float, device = sycl_device)
# index = torch.tensor([0], device = sycl_device)
# x_sycl.index_add_(1, index, t)
# print("x_sycl = ", x_sycl.to("cpu"))
