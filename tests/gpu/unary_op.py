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

print("erf cpu", torch.erf(y_cpu_float))
print("erf dpcpp", torch.erf(y_dpcpp_float).cpu())

print("erfc cpu", torch.erfc(y_cpu_float))
print("erfc dpcpp", torch.erfc(y_dpcpp_float).cpu())

print("exp cpu", torch.exp(y_cpu_float))
print("exp dpcpp", torch.exp(y_dpcpp_float).cpu())

print("log cpu", torch.log(y_cpu_float))
print("log dpcpp", torch.log(y_dpcpp_float).cpu())

print("log10 cpu", torch.log10(y_cpu_float))
print("log10 dpcpp", torch.log10(y_dpcpp_float).cpu())

print("log1p cpu", torch.log1p(y_cpu_float))
print("log1p dpcpp", torch.log1p(y_dpcpp_float).cpu())

print("log2 cpu", torch.log2(y_cpu_float))
print("log2 dpcpp", torch.log2(y_dpcpp_float).cpu())

print("rsqrt cpu", torch.rsqrt(y_cpu_float))
print("rsqrt dpcpp", torch.rsqrt(y_dpcpp_float).cpu())

print("sqrt cpu", torch.sqrt(y_cpu_float))
print("sqrt dpcpp", torch.sqrt(y_dpcpp_float).cpu())

print("__and__ y_cpu", y_cpu_int8.__and__(3))
print("__and__ y_dpcpp", y_dpcpp_int8.__and__(3).to("cpu"))

print("__iand__ y_cpu", y_cpu_int8.__iand__(3))
print("__iand__ y_dpcpp", y_dpcpp_int8.__iand__(3).to("cpu"))

print("__or__ y_cpu", y_cpu_int8.__or__(3))
print("__or__ y_dpcpp", y_dpcpp_int8.__or__(3).to("cpu"))

print("__ior__ y_cpu", y_cpu_int8.__ior__(3))
print("__ior__ y_dpcpp", y_dpcpp_int8.__ior__(3).to("cpu"))

print("erfinv cpu ", torch.erfinv(y_cpu_float))
print("erfinv dpcpp", torch.erfinv(y_dpcpp_float.to("cpu")))

print("digamma cpu ", torch.digamma(y_cpu_float))
print("digamma dpcpp ", torch.digamma(y_dpcpp_float.to("cpu")))

print("sign cpu", torch.sign(y_cpu_float));
print("sign dpcpp", torch.sign(y_dpcpp_float).to("cpu"));
