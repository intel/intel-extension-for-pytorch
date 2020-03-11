import torch
import torch_ipex

cpu_device = torch.device("cpu")
x_cpu1 = torch.randn((5, 4))
x_cpu2 = torch.randn((5, 4))
x_dpcpp1 = x_cpu1.to("dpcpp")
x_dpcpp2 = x_cpu2.to("dpcpp")

print("Before:")
print("self dpcpp", x_dpcpp1.to("cpu"));
print("src dpcpp", x_dpcpp2.to("cpu"));

x_dpcpp1.set_(x_dpcpp2)

print("After:")
print("self dpcpp", x_dpcpp1.to("cpu"));
print("src dpcpp", x_dpcpp2.to("cpu"));
