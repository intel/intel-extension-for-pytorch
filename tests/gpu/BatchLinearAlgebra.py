import torch
import torch_ipex

cpu_device = torch.device("cpu")
##sycl_device = torch.device("sycl")
#x_cpu = torch.tensor((), dtype=torch.int32, device = cpu_device)
x_cpu = torch.randn(5,5)

x_sycl = x_cpu.to("dpcpp")
#y_cpu1 = x_cpu.new_ones((2, 3))
y_cpu1 = torch.randn(5,5)
#y_cpu2 = x_cpu.new_ones((2, 3))
y_cpu2 = torch.randn(5,5)

y_sycl1 = y_cpu1.to("dpcpp")
y_sycl2 = y_cpu2.to("dpcpp")

print("y_cpu", torch.tril(y_cpu2))
print("y_sycl", torch.tril(y_sycl2).to("cpu"))

print("y_cpu", torch.triu(y_cpu2))
print("y_sycl", torch.triu(y_sycl2).to("cpu"))