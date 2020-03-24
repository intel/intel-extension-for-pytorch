import torch
import numpy as np
import torch_ipex
np.set_printoptions(threshold=np.inf)

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

x_cpu = torch.randn([3,3], dtype=torch.float, device = cpu_device)
y_cpu = torch.randn([3,3], dtype=torch.float, device = cpu_device)
mask_cpu = y_cpu.gt(0)
print("x_cpu:")
print(x_cpu)
print("mask_cpu:")
print(mask_cpu)
print("x_cpu[mask_cpu]:")
print(x_cpu[mask_cpu])

#dpcpp part
x_dpcpp = x_cpu.to("dpcpp")
mask_dpcpp = mask_cpu.to("dpcpp")
print("mask index:")
print(mask_dpcpp.nonzero().to("cpu"))
print("x_dpcpp[mask_dpcpp]:")
print(x_dpcpp[mask_dpcpp].to("cpu"))

#index put
input = torch.ones([1], dtype=torch.float, device = cpu_device)
indcies = torch.tensor([0, 0])
x_cpu[indcies] = input
print("index_put")
print(x_cpu)
x_cpu.index_put_([indcies], input, True)
print("index_put accmulate=true")
print(x_cpu)

input = input.to("dpcpp")
indcies = indcies.to("dpcpp")
x_dpcpp[indcies] = input
print("dpcpp index_put")
print(x_dpcpp.cpu())
x_dpcpp.index_put_([indcies], input, True)
print("dpcpp  index_put accmulate=true")
print(x_dpcpp.cpu())
