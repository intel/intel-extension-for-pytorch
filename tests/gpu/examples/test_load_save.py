import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa
import tempfile

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TheModelClass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TheModelClass, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        x = torch.log(self.linear(x))
        return x

class TestTorchMethod(TestCase):
    def test_load_save(self, dtype=torch.float):
        save = torch.randn((2, 2), device='xpu', dtype=dtype)
        ckpt = tempfile.NamedTemporaryFile()
        torch.save(save, ckpt.name)
        load = torch.load(ckpt.name, map_location=xpu_device)
        self.assertEqual(save.to(cpu_device), load.to(cpu_device))

        torch.save(save, ckpt.name, _use_new_zipfile_serialization=False)
        load = torch.load(ckpt.name, map_location=xpu_device)
        self.assertEqual(save.to(cpu_device), load.to(cpu_device))

        save = TheModelClass(2, 3).to('xpu')
        torch.save(save, ckpt.name)
        load = torch.load(ckpt.name, map_location=xpu_device)
        self.assertEqual(save.to(cpu_device).__str__(), load.to(cpu_device).__str__())

        torch.save(save, ckpt.name, _use_new_zipfile_serialization=False)
        load = torch.load(ckpt.name, map_location=xpu_device)
        self.assertEqual(save.to(cpu_device).__str__(), load.to(cpu_device).__str__())
