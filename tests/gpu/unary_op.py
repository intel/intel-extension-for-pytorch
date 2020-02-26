import torch
import torch_ipex

cpu_device = torch.device("cpu")

y_cpu_float = torch.randn([2,2,2,2], device=cpu_device, dtype=torch.float)
y_cpu_int8 = torch.tensor([-1, 1], device=cpu_device, dtype=torch.int8)
y_dpcpp_float = y_cpu_float.to("dpcpp")
y_dpcpp_int8 = y_cpu_int8.to("dpcpp")

# print("-cpu (neg)", -y_cpu_float)
# print("-dpcpp (neg)", (-y_dpcpp_float).cpu())
# 
# print("neg cpu", torch.neg(y_cpu_float))
# print("neg dpcpp", torch.neg(y_dpcpp_float).cpu())
# 
# print("bitwise_not cpu", torch.bitwise_not(y_cpu_int8))
# print("bitwise_not dpcpp", torch.bitwise_not(y_dpcpp_int8).cpu())
# 
# print("logical_not cpu", torch.logical_not(y_cpu_int8))
# print("logical_not dpcpp", torch.logical_not(y_dpcpp_int8).cpu())
# 
# print("acos cpu", torch.acos(y_cpu_float))
# print("acos dpcpp", torch.acos(y_dpcpp_float).cpu())
# 
# print("asin cpu", torch.asin(y_cpu_float))
# print("asin dpcpp", torch.asin(y_dpcpp_float).cpu())
# 
# print("ceil cpu", torch.ceil(y_cpu_float))
# print("ceil dpcpp", torch.ceil(y_dpcpp_float).cpu())
# 
# print("expm1 cpu", torch.expm1(y_cpu_float))
# print("expm1 dpcpp", torch.expm1(y_dpcpp_float).cpu())
# 
# print("round cpu", torch.round(y_cpu_float))
# print("round dpcpp", torch.round(y_dpcpp_float).cpu())
# 
# print("trunc cpu", torch.trunc(y_cpu_float))
# print("trunc dpcpp", torch.trunc(y_dpcpp_float).cpu())

print("clamp cpu", torch.clamp(y_cpu_float, min=-0.1, max=0.5))
print("clamp dpcpp ", torch.clamp(y_dpcpp_float, min=-0.1, max=0.5).cpu())
