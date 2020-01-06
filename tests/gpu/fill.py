import torch
import torch_ipex

cpu_device =  torch.device("cpu")
sycl_device =  torch.device("dpcpp")
# x = torch.ones([1, 2, 3, 4], device=cpu_device)
# y = x.fill_(2)
x = torch.ones([1, 2, 3, 4], device=sycl_device)
y = x.fill_(2)
# print(y_cpu)

# x_sycl1 = x_cpu1.to("sycl")
# 
# 
# y_sycl = x_sycl1[1].fill_(2)
# print("sycl:", y_sycl.cpu())
