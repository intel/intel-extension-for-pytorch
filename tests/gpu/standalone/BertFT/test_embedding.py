import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
 
import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

shapes = [
        ((2, 1024), (2, 384)),
        ((512, 1024), (1, 384)),
        ((30522, 1024), (2, 384))
]

class TestTorchMethod(TestCase):
    def test_embedding(self, dtype=torch.float):
        for shape in shapes:
            input = torch.ones(shape[1]).long()
            weight = torch.randn(shape[0])
            input_xpu = input.xpu()
            weight_xpu = weight.xpu()
            output = F.embedding(input, weight)
            output_xpu = F.embedding(input_xpu, weight_xpu)
            self.assertEqual(output, output_xpu.cpu())
