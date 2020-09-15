import torch
import torch_ipex
import torch.nn as nn

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class M(torch.nn.Module):
    def forward(self, a, b, c):
        o = a * b
        o += c
        return o


a = torch.randn(1, 2, 3, 3, dtype=torch.float).to("dpcpp")
b = torch.randn(1, 2, 3, 3, dtype=torch.float).to("dpcpp")
c = torch.randn(1, 2, 3, 3, dtype=torch.float).to("dpcpp")

o = a * b + c
print("eager: ", o.cpu())

model = M().eval()
m = torch.jit.script(model.eval().to("dpcpp"))
with torch.no_grad():
    # print(m.graph_for(a, b, c))
    print(m(a, b, c).cpu())
