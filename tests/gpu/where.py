import torch
import torch_ipex

x = torch.tensor([[ 0.6580, -1.0969, -0.4614], [-0.1034, -0.5790,  0.1497]])
x_ones = torch.tensor([[ 1., 1., 1.], [1.,1.,1.]])
print("cpu",torch.where(x > 0, x, x_ones))

x_dpcpp = torch.tensor([[ 0.6580, -1.0969, -0.4614], [-0.1034, -0.5790,  0.1497]], device = torch.device("dpcpp"))
x_ones_dpcpp = torch.tensor([[ 1., 1., 1.], [1., 1., 1.]], device = torch.device("dpcpp"))
print("dpcpp",torch.where(x_dpcpp > 0, x_dpcpp, x_ones_dpcpp).cpu())
