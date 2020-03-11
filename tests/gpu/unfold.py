import torch
import torch_ipex

x_cpu =  torch.tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
y = x_cpu.unfold(0, 2, 1)
x_dpcpp = torch.tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.], device = torch.device("dpcpp"))
y_dpcpp = x_dpcpp.unfold(0, 2, 1)
print("unfold cpu ", y)
print("unfold dpcpp ", y_dpcpp.to("cpu"))
