import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")
checking_atol = 1e-2
checking_rtol = 3e-2

class TestNet(torch.nn.Module):
    __test__ = False

    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        return x


class TestTorchMethod(TestCase):
    def test_autocast_simple_forward_bf16(self):
        model = TestNet()
        x = torch.ones([2, 3, 8, 6], dtype=torch.float)

        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            y = model(x)

        model.to('xpu')
        x_xpu = x.to('xpu')
        with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            y_xpu = model(x_xpu)

        self.assertEqual(y, y_xpu.to('cpu'))

    def test_autocast_simple_forward_fp16(self):
        # cpu now does not support fp16 for autocast
        model = TestNet()
        x = torch.ones([2, 3, 8, 6], dtype=torch.float)

        model.to('xpu')
        x_xpu = x.to('xpu')
        with torch.autocast(device_type='xpu', enabled=True, dtype=torch.float16):
            y_xpu = model(x_xpu)

        print(y_xpu.to('cpu'))
        self.assertEqual(y_xpu.dtype, torch.float16)

    def test_autocast_simple_backward_bf16(self):
        model = TestNet()
        x = torch.ones([2, 3, 8, 6], dtype=torch.float)
        x_xpu = x.to('xpu')
        x.requires_grad_(True)
        x_xpu.requires_grad_(True)

        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            y = model(x)
            loss = y.sum()
        loss.backward()

        gw_conv1 = model.conv1.weight.grad.clone()
        gw_conv2 = model.conv2.weight.grad.clone()

        model.zero_grad(set_to_none=True)
        model.to('xpu')
        with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            y_xpu = model(x_xpu)
            loss = y_xpu.sum()
        loss.backward()

        gw_conv1_xpu = model.conv1.weight.grad
        gw_conv2_xpu = model.conv2.weight.grad

        self.assertEqual(y, y_xpu.to('cpu'))
        self.assertEqual(gw_conv2, gw_conv2_xpu.to('cpu'))
        self.assertEqual(gw_conv1, gw_conv1_xpu.to('cpu'), atol=checking_atol, rtol=checking_rtol)
        self.assertEqual(x.grad, x_xpu.grad.to('cpu'), atol=checking_atol, rtol=checking_rtol)
