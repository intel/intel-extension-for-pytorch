import torch

a = torch.randn(2, 3, 2)
print(a.size())
print(a)
x, y = torch.min(a, -1)
print(x.size(), y.size())
print(x, y)
