import torch
import time
import torch_ipex

x_cpu1 = torch.tensor([[1., 2.], [3., 4.]])
x_cpu2 = torch.tensor([[1., 1.], [4., 4.]])
x_cpu3 = torch.tensor([[1., 2.], [3., 4.]])
y_cpu = torch.eq(x_cpu1, x_cpu2)
print("eq cpu", y_cpu)
print("equal cpu1", torch.equal(x_cpu1, x_cpu2))
print("equal cpu2", torch.equal(x_cpu1, x_cpu3))

x_dpcpp1 = x_cpu1.to("dpcpp")
x_dpcpp2 = x_cpu2.to("dpcpp")
x_dpcpp3 = x_cpu3.to("dpcpp")

y_dpcpp = torch.eq(x_dpcpp1, x_dpcpp2)
print("eq dpcpp", y_dpcpp.cpu())
print("eqeual dpcpp1", torch.equal(x_dpcpp1, x_dpcpp2))
print("eqeual dpcpp2", torch.equal(x_dpcpp1, x_dpcpp3))
