import torch
import torch_ipex

a = torch.ones([10], dtype=torch.float64)
a = a.to("xpu")
b=a.storage()
print(b)
print("save")
torch.save(a, './a_tensor.pt')

print("load")
c = torch.load('./a_tensor.pt')
d=c.storage()
print(d)