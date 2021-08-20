import torch
import torch.nn as nn
import torch.nn.functional as F
import ipex


class Conv2dRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        return F.relu(self.conv(x) + a, inplace=True)


a1 = torch.ones([1, 2, 1, 1])
a1.fill_(2)

model = Conv2dRelu(2, 2, kernel_size=3, stride=1, bias=True) # // you can find a simple model in tests/example/test_fusion.py

print("johnlu module to ")
model = model.to('xpu').eval()
input = torch.randn([1, 2, 3, 3])  # torch.randn((conv_input_shape)).to(“xpu”)

print("johnlu to")
a1 = a1.to('xpu')
input = input.to('xpu')

modelJit = torch.jit.script(model)
# modelJit = torch.jit.trace(model, (input, a1))
for param in modelJit.parameters():
    print("johnlu original modelJit param", param.cpu())

modelJit.save('./simple_trace_case.zip')
modelJit = torch.jit.load('./simple_trace_case.zip')

for param in modelJit.parameters():
    print("johnlu loaded modelJit param", param.cpu())

