import torch
import torch_ipex

cpu_device = torch.device("cpu")

y_cpu_float = torch.randn([2,2,2,2], device=cpu_device, dtype=torch.float)
y_cpu_int8 = torch.tensor([-1, 1], device=cpu_device, dtype=torch.int8)
y_dpcpp_float = y_cpu_float.to("dpcpp")
y_dpcpp_int8 = y_cpu_int8.to("dpcpp")

print("-cpu (neg)", -y_cpu_float)
print("-dpcpp (neg)", (-y_dpcpp_float).cpu())

print("neg cpu", torch.neg(y_cpu_float))
print("neg dpcpp", torch.neg(y_dpcpp_float).cpu())

print("bitwise_not cpu", torch.bitwise_not(y_cpu_int8))
print("bitwise_not dpcpp", torch.bitwise_not(y_dpcpp_int8).cpu())

print("logical_not cpu", torch.logical_not(y_cpu_int8))
print("logical_not dpcpp", torch.logical_not(y_dpcpp_int8).cpu())

print("cos cpu", torch.cos(y_cpu_float))
print("cos dpcpp", torch.cos(y_dpcpp_float).cpu())

print("sin cpu", torch.sin(y_cpu_float))
print("sin dpcpp", torch.sin(y_dpcpp_float).cpu())

print("tan cpu", torch.tan(y_cpu_float))
print("tan dpcpp", torch.tan(y_dpcpp_float).cpu())

print("cosh cpu", torch.cosh(y_cpu_float))
print("cosh dpcpp", torch.cosh(y_dpcpp_float).cpu())

print("sinh cpu", torch.sinh(y_cpu_float))
print("sinh dpcpp", torch.sinh(y_dpcpp_float).cpu())

print("tanh cpu", torch.tanh(y_cpu_float))
print("tanh dpcpp", torch.tanh(y_dpcpp_float).cpu())

print("acos cpu", torch.acos(y_cpu_float))
print("acos dpcpp", torch.acos(y_dpcpp_float).cpu())

print("asin cpu", torch.asin(y_cpu_float))
print("asin dpcpp", torch.asin(y_dpcpp_float).cpu())

print("atan cpu", torch.atan(y_cpu_float))
print("atan dpcpp", torch.atan(y_dpcpp_float).cpu())

print("ceil cpu", torch.ceil(y_cpu_float))
print("ceil dpcpp", torch.ceil(y_dpcpp_float).cpu())

print("expm1 cpu", torch.expm1(y_cpu_float))
print("expm1 dpcpp", torch.expm1(y_dpcpp_float).cpu())

print("round cpu", torch.round(y_cpu_float))
print("round dpcpp", torch.round(y_dpcpp_float).cpu())

print("trunc cpu", torch.trunc(y_cpu_float))
print("trunc dpcpp", torch.trunc(y_dpcpp_float).cpu())

print("clamp cpu", torch.clamp(y_cpu_float, min=-0.1, max=0.5))
print("clamp dpcpp ", torch.clamp(y_dpcpp_float, min=-0.1, max=0.5).cpu())
