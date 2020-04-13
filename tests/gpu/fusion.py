import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_ipex


class Conv2dRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return F.relu(self.conv(x) + a, inplace=True)

x = torch.randn([1, 2, 3, 3], device=torch.device("dpcpp"), dtype=torch.float)
a1 = torch.ones([1, 2, 1, 1], device=torch.device("dpcpp"), dtype=torch.float)
a2 = torch.ones([1, 2, 1, 1], device=torch.device("dpcpp"), dtype=torch.float)
model = Conv2dRelu(2, 2, kernel_size = 3, stride = 1, bias=False)
model.to("dpcpp")
y = model(x, a1)
print("raw: ", y.cpu())

modelJit = torch.jit.script(model)
modelJit.to("dpcpp")
print(modelJit.graph)
with torch.no_grad():
    # print(modelJit.graph_for(x))
    print("fusion:", modelJit(x, a2).cpu())

del modelJit
