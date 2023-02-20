from collections import OrderedDict
import time
import torch
import torch.nn as nn
import intel_extension_for_pytorch
import cosim

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 3, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = x * 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x, x

class CompositeModule(nn.Module):
    def  __init__(self):
        super().__init__()
        self.m1 = BaseModule()
        self.m2 = BaseModule()

    def forward(self, x, bias=torch.ones(1).xpu()):
        x = x + x
        x1, x2 = self.m1(x)
        x = x1 + x2 + bias
        x1, x2 = self.m2(x)
        x = x1 + x2 + bias
        return x * 2

class SequentialModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
                ('c1', CompositeModule()),
                ('c2', CompositeModule())
            ]
        ))

    def forward(self, x):
        x = self.model(x)
        # x = self.fc(x)
        return x

class ModuleListModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.modellist = nn.ModuleList([
            SequentialModule(),
            SequentialModule()
        ])

    def forward(self, x):
        y = None
        for layer in self.modellist[:]:
            x = layer(x)
        return x

if __name__ == '__main__':
    model = ModuleListModule().to('xpu')
    cosim_model = cosim.CosimModule(model)
    a = torch.randn(2, 3, 16, 16, requires_grad=True, device='xpu', dtype=torch.float32)
    for epoch in range(5):
        bc = cosim_model(a)
        bc.backward(torch.ones_like(bc))
        cosim_model.plot_result(file='cosim_outputs/1/')
        a = torch.randn(2, 3, 16, 16, requires_grad=True, device='xpu', dtype=torch.float32)
