import torch
import intel_extension_for_pytorch # noqa
from torch.testing._internal.common_utils import TestCase
import torch.nn as nn
import pytest

class TestNet(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(TestNet, self).__init__()
        self.norm = nn.GroupNorm(8, 16, affine=True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False)
        self.s = nn.Sequential(self.conv, self.norm)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class TestTorchMethod(TestCase):
    # @pytest.mark.skipif(torch.xpu.using_onednn_layout(), reason="channels last does not support onednn block format")
    @pytest.mark.skip(reason="Wrong results in Conv2d channels_last in the shape")
    def test_group_norm_forward(self, dtype=torch.float):
        x = torch.randn(16, 16, 64, 64, requires_grad=True, device=torch.device('cpu'))
        x_dpcpp = x.to(memory_format=torch.channels_last).to("xpu")

        model = TestNet(16, 16)

        model.eval()
        y_pred = model(x)
        model_dpcpp = model.to(torch.device('xpu'))
        model_dpcpp.eval()
        y_pred_dpcpp = model_dpcpp(x_dpcpp)
        self.assertEqual(y_pred_dpcpp.is_contiguous(memory_format=torch.channels_last), True)
        self.assertEqual(y_pred, y_pred_dpcpp)

    # @pytest.mark.skipif(torch.xpu.using_onednn_layout(), reason="channels last does not support onednn block format")
    @pytest.mark.skip(reason="Wrong results in Conv2d channels_last in the shape")
    def test_group_norm_backward(self, dtype=torch.float):
        x = torch.randn(16, 16, 64, 64, requires_grad=True, device=torch.device('cpu'))
        x_dpcpp = x.to(memory_format=torch.channels_last).to("xpu")

        for model in [TestNet(16, 16), nn.GroupNorm(8, 16, affine=True)]:
            model.eval()
            y_pred = model(x)
            grad_out = torch.randn_like(y_pred)
            grad, = torch.autograd.grad(y_pred, x, grad_out)

            model_dpcpp = model.to(torch.device('xpu'))
            model_dpcpp.eval()
            y_pred_dpcpp = model_dpcpp(x_dpcpp)
            grad_dpcpp, = torch.autograd.grad(y_pred_dpcpp, x_dpcpp, grad_out.to("xpu"))
            self.assertEqual(grad_dpcpp.is_contiguous(memory_format=torch.channels_last), True)
            self.assertEqual(grad, grad_dpcpp.cpu())
