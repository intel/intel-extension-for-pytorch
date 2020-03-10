import torch
import torch_ipex

x1 = torch.randn(10, device=torch.device("cpu"))
x1_dpcpp = x1.to("dpcpp")
print("(10) cpu", torch.cumsum(x1, dim=0))
print("(10) dpcpp", torch.cumsum(x1_dpcpp, dim=0).cpu())

x1_dpcpp = x1.to("dpcpp").to(torch.float16)
# print("(10) half cpu", torch.cumsum(x1, dim=0))
print("(10) half dpcpp", torch.cumsum(x1_dpcpp, dim=0).cpu())

x2 = torch.randn(3, 2, 4, device=torch.device("cpu"))
x2_dpcpp = x2.to("dpcpp")
print("(3, 2, 4) cpu", torch.cumsum(x2, dim=0))
print("(3, 2, 4) dpcpp", torch.cumsum(x2_dpcpp, dim=0).cpu())

x3 = torch.randn(4, 2, 4, device=torch.device("cpu"))
x3_dpcpp = x3.to("dpcpp")
print("(4, 2, 4) cpu", torch.cumsum(x3, dim=2))
print("(4, 2, 4) dpcpp", torch.cumsum(x3_dpcpp, dim=2).cpu())
