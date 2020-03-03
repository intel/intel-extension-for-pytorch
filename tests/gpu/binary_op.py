import torch
import torch_ipex

cpu_device = torch.device("cpu")
##sycl_device = torch.device("sycl")
#x_cpu = torch.tensor((), dtype=torch.int32, device = cpu_device)
x_cpu = torch.randn(5)

x_sycl = x_cpu.to("dpcpp")
#y_cpu1 = x_cpu.new_ones((2, 3))
y_cpu1 = torch.randn(5)
#y_cpu2 = x_cpu.new_ones((2, 3))
y_cpu2 = torch.randn(5)

y_cpu1_int = torch.tensor([[3, 1, 2, 3], [2, 3, 4, 1]], dtype=torch.int32)
#y_cpu2 = x_cpu.new_ones((2, 3))
y_cpu2_int = torch.tensor([[1, 5, 2, 4], [1, 1, 5, 5]], dtype=torch.int32)

y_sycl1 = y_cpu1.to("dpcpp")
y_sycl2 = y_cpu2.to("dpcpp")
y_sycl1_int = y_cpu1_int.to("dpcpp")
y_sycl2_int = y_cpu2_int.to("dpcpp")

# print("add y_cpu", y_cpu1.add(y_cpu2))
# print("add y_sycl", y_sycl1.add(y_sycl2).to("cpu"))
# 
# print("sub y_cpu", y_cpu1.sub(y_cpu2))
# print("sub y_sycl", y_sycl1.sub(y_sycl2).to("cpu"))
# 
# print("mul y_cpu", y_cpu1.mul(y_cpu2))
# print("mul y_sycl", y_sycl1.mul(y_sycl2).to("cpu"))
# 
# print("div y_cpu", y_cpu1.div(y_cpu2))
# print("div y_sycl", y_sycl1.div(y_sycl2).to("cpu"))

print("__and__ y_cpu", y_cpu1_int.__and__(y_cpu2_int))
print("__and__ y_sycl", y_sycl1_int.__and__(y_sycl2_int).to("cpu"))

print("__iand__ y_cpu", y_cpu1_int.__iand__(y_cpu2_int))
print("__iand__ y_sycl", y_sycl1_int.__iand__(y_sycl2_int).to("cpu"))

print("__or__ y_cpu", y_cpu1_int.__or__(y_cpu2_int))
print("__or__ y_sycl", y_sycl1_int.__or__(y_sycl2_int).to("cpu"))

print("__ior__ y_cpu", y_cpu1_int.__ior__(y_cpu2_int))
print("__ior__ y_sycl", y_sycl1_int.__ior__(y_sycl2_int).to("cpu"))
