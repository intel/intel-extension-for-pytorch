import torch
import torch_ipex

x_cpu = torch.randn(2, 3)
x_dpcpp = x_cpu.to("dpcpp")

print("x_cpu", x_cpu, " median_cpu", torch.median(x_cpu))
print("x_dpcpp", x_dpcpp.to("cpu"), " median_dpcpp", torch.median(x_dpcpp).to("cpu"))


x_cpu2 = torch.tensor(([1,2,3,4,5]),dtype=torch.int32, device=torch.device("cpu"))
x_dpcpp2 = torch.tensor(([1,2,3,4,5]),dtype=torch.int32, device=torch.device("dpcpp"))

print("x_cpu2", x_cpu2, " median_cpu2", x_cpu2.median())
print("x_dpcpp2", x_dpcpp2.to("cpu"), " median_dpcpp2", x_dpcpp2.median().to("cpu"))
