import torch
import torch_ipex

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
x.narrow(0, 0, 2)
x_dpcpp = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device = torch.device("dpcpp"))
x_dpcpp.narrow(0, 0, 2)
print("x = ",x)
print("x_dpcpp = ", x_dpcpp.to("cpu"))
