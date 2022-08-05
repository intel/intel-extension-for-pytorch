import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm1d(3)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class TestNNMethod(TestCase):
    def test_model_conversion_channels_last_1d(self, dtype=torch.float):
        model = Model()
        test_input = torch.rand([2, 3, 4])
        model = model.to(cpu_device)
        cpu_res = model(test_input)

        test_input_xpu = test_input.to(dpcpp_device)
        model = model.to(dpcpp_device)
        model = torch.xpu.to_channels_last_1d(model)
        xpu_res = model(test_input_xpu)

        self.assertEqual(torch.xpu.is_contiguous_channels_last_1d(xpu_res), True)
        self.assertEqual(xpu_res.cpu(), cpu_res)
