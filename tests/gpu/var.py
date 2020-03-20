import torch
import torch_ipex

src = torch.randn((3,4,), dtype=torch.float32, device = torch.device("cpu"))
print("cpu src = ", src)
print("cpu dst = ", src.var())
print("cpu dst with dim = ", src.var(1))

src_dpcpp = src.to("dpcpp")
print("gpu src = ", src_dpcpp.cpu())
print("gpu dst = ", src_dpcpp.var().cpu())
print("gpu dst with dim = ", src_dpcpp.var(1).cpu())
