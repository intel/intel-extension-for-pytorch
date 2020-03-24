import torch
import torch_ipex

a = torch.randn((4, 3), device = torch.device("cpu"))
b = torch.randn((4, 3), device = torch.device("cpu"))

print(a.cross(b))

a_dpcpp = a.to("dpcpp")
b_dpcpp = b.to("dpcpp")
print(a_dpcpp.cross(b_dpcpp).cpu())
