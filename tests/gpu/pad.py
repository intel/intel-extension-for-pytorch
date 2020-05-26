import torch
import torch_ipex

F = torch.nn.functional

print("cpu")
t4d = torch.empty(3, 3, 4, 2)
p1d = (1, 1) # pad last dim by 1 on each side
out = F.pad(t4d, p1d, "constant", 0)
print(out.size())
t4d = torch.empty(3, 3, 4, 2)
p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
out = F.pad(t4d, p2d, "constant", 0)
print(out.size())
t4d = torch.empty(3, 3, 4, 2)
p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
out = F.pad(t4d, p3d, "constant", 0)
print(out.size())

print("sycl")
t4d = torch.empty(3, 3, 4, 2).to("dpcpp")
p1d = (1, 1) # pad last dim by 1 on each side
out = F.pad(t4d, p1d, "constant", 0)
print(out.size())
t4d = torch.empty(3, 3, 4, 2).to("dpcpp")
p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
out = F.pad(t4d, p2d, "constant", 0)
print(out.size())
t4d = torch.empty(3, 3, 4, 2).to("dpcpp")
p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
out = F.pad(t4d, p3d, "constant", 0)
print(out.size())
