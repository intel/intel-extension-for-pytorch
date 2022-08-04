import tempfile
import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch # noqa


class TestTorchMethod(TestCase):
    def test_save_load(self):
        a = torch.ones([10], dtype=torch.float64)
        a = a.to("xpu")
        ckpt = tempfile.NamedTemporaryFile()
        torch.save(a, ckpt.name)
        b = torch.load(ckpt.name)
        assert torch.equal(a, b), "tensor saved & loaded not equal"
