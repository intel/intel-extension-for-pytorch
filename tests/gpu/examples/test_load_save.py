import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch
import tempfile
import pytest

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_load_save(self, dtype=torch.float):
        save = torch.randn(2, 2)
        print("save: ", save)
        save = save.to('xpu')
        ckpt = tempfile.NamedTemporaryFile()
        torch.save(save, ckpt.name)

        load = torch.load(ckpt.name, map_location=xpu_device)
        print("torch.load(ckpt.name, map_location=xpu_device): ", load.to(cpu_device))

        self.assertEqual(save.to(cpu_device), load.to(cpu_device))
