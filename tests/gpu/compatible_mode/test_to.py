import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch as ipex  # noqa
ipex.compatible_mode()

cuda_device = torch.device("cuda")


class TestTorchMethod(TestCase):
    def test_to(self):
        x = torch.empty(4, 5).to("cuda:0")
        self.assertEqual(x.device.type, "xpu")

    def test_cuda_str(self):
        x = torch.empty(4, 5).cuda("cuda:0")
        self.assertEqual(x.device.type, "xpu")

    def test_cuda_int(self):
        n = torch.cuda.device_count()
        x = torch.empty(4, 5).cuda(n - 1)
        self.assertEqual(x.device.type, "xpu")

    def test_cuda_none(self):
        x = torch.empty(4, 5).cuda()
        self.assertEqual(x.device.type, "xpu")

    def test_pin_memory(self):
        x = torch.randn(3, 3)
        x = x.pin_memory(device='cuda')
        
        x = torch.randn(3, 3)
        x = x.pin_memory('cuda')

        x = torch.randn(3, 3)
        x = x.pin_memory()
    
    def test_is_cuda(self):
        x = torch.randn(3, 3, device='cuda')
        self.assertEqual(x.is_xpu, True)
    
    def test_nn_model_cuda(self):
        class toy_model(nn.Module):
            def __init__(self):
                super(toy_model, self).__init__()
                self.m = nn.Sequential(
                nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
                )
                self.fc = nn.Linear(in_features=2048, out_features=1000)

            def forward(self, x):
                x = self.m(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = toy_model().cuda()
        for param in model.parameters():
            self.assertEqual(param.device.type, "xpu")

    def test_cuda(self):
        item = torch.rand(1, 2, 3).cuda()
        self.assertEqual(item.device.type, "xpu")

        item_index = torch.rand(1, 2, 3).cuda(0)
        self.assertEqual(item_index.device.type, "xpu")
        self.assertEqual(item_index.device.index, 0)
