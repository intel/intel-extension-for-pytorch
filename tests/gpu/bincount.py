import torch
import torch_ipex

dev="dpcpp"

x_cpu = torch.randint(0, 8, (5,), dtype=torch.int64)
y_cpu = torch.linspace(0, 1, steps=5)
x_dpcpp = x_cpu.to(dev)
y_dpcpp = y_cpu.to(dev)


print("bincount cpu 1", torch.bincount(x_cpu))
# print("bincount cpu 2" x_cpu.bincount(y_cpu))
print("bincount dpcpp 1", torch.bincount(x_dpcpp).cpu())
# print("bincount dpcpp 2" x_dpcpp.bincount(y_dpcpp).cpu())
