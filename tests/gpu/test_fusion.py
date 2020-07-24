import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import pytest


torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class Conv2dRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x, a):
        # return F.relu(self.conv(x), inplace=True)
        return F.relu(self.conv(x) + a, inplace=True)
        # return self.conv(x) + a


class TestNNMethod(TestCase):
    @pytest.mark.skip(reason='le-5 blocked')
    def test_fusion(self, dtype=torch.float):
        x = torch.randn([1, 2, 3, 3], device=cpu_device)
        a1 = torch.ones([1, 2, 1, 1], device=cpu_device)
        a2 = torch.ones([1, 2, 1, 1], device=dpcpp_device)
        a3 = torch.ones([1, 2, 1, 1], device=dpcpp_device)

        a1.fill_(2)
        a3.fill_(2)

        model = Conv2dRelu(2, 2, kernel_size=3, stride=1, bias=True)
        y = model(x, a1)
        print("raw: ", y)

        x = x.to("dpcpp")
        model.to("dpcpp")
        modelJit = torch.jit.script(model)
        # modelJit.to("dpcpp")
        # print(modelJit.graph)
        with torch.no_grad():
            # print(modelJit.graph_for(x, a2))
            print("fusion:", modelJit(x, a3).cpu())
            y_dpcpp = modelJit(x, a3)
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        del modelJit
