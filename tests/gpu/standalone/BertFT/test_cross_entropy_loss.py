import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
 
import intel_extension_for_pytorch  # noqa

shapes = [
        (1, 1000),
        (2, 384)
]

class TestTorchMethod(TestCase):
    def test_cross_entropy(self, dtype=torch.float):
        for shape in shapes:
            input = torch.randn(shape[0], shape[1], requires_grad=True)
            #target = torch.randn(3, 5).softmax(dim=1)
            target = torch.randint(shape[1], (shape[0],), dtype=torch.int64)
    
            loss = F.cross_entropy(input, target)
            #print("cpu result: ", loss)
    
            loss_xpu = F.cross_entropy(input.xpu(), target.xpu())
            #print("xpu result: ", loss_xpu.cpu())
    
            self.assertEqual(loss, loss_xpu.cpu())
