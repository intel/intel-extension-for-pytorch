import torch

import torch_ipex
x = torch.randn(3, 4, dtype=torch.float, device=torch.device("cpu"))
x_mask = x.ge(0.5)

print("x",x)
print("mask", x_mask)
print("cpu masked_select", torch.masked_select(x, x_mask))

y = x.to("dpcpp")
y_mask = x_mask.to("dpcpp")
print("sycl masked_select", torch.masked_select(y, y_mask).cpu())

