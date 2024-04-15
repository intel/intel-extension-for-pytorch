import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
 
import intel_extension_for_pytorch  # noqa

class TestTorchMethod(TestCase):
    def test_cross_entropy_1(self, dtype=torch.float):
        input = torch.randn((1, 2), requires_grad=True, dtype=dtype)
        target = torch.randint(2, (1,), dtype=torch.int64)
        loss = F.cross_entropy(input, target)
        #print("cpu result: ", loss)
        loss_xpu = F.cross_entropy(input.xpu(), target.xpu())
        #print("xpu result: ", loss_xpu.cpu())
        self.assertEqual(loss, loss_xpu.cpu())

    def test_cross_entropy_bfloat16_1(self, dtype=torch.bfloat16):
        input = torch.randn((1, 2), requires_grad=True, dtype=dtype)
        target = torch.randint(2, (1,), dtype=torch.int64)
        loss = F.cross_entropy(input, target)
        #print("cpu result: ", loss)
        loss_xpu = F.cross_entropy(input.xpu(), target.xpu())
        #print("xpu result: ", loss_xpu.cpu())
        self.assertEqual(loss, loss_xpu.cpu())

    def test_cross_entropy_float16_1(self, dtype=torch.bfloat16):
        input = torch.randn((1, 2), requires_grad=True, dtype=dtype)
        target = torch.randint(2, (1,), dtype=torch.int64)
        loss = F.cross_entropy(input, target)
        #print("cpu result: ", loss)
        dtype_dpcpp = torch.float16
        loss_xpu = F.cross_entropy(input.xpu().to(dtype_dpcpp), target.xpu())
        #print("xpu result: ", loss_xpu.cpu())
        self.assertEqual(loss, loss_xpu.cpu().to(torch.bfloat16))

    def test_cross_entropy_2(self, dtype=torch.float):
        input = torch.randn((512, 30522), requires_grad=True, dtype=dtype)
        target = torch.randint(30522, (512,), dtype=torch.int64)
        loss = F.cross_entropy(input, target)
        #print("cpu result: ", loss)
        loss_xpu = F.cross_entropy(input.xpu(), target.xpu())
        #print("xpu result: ", loss_xpu.cpu())
        self.assertEqual(loss, loss_xpu.cpu())

    def test_cross_entropy_bfloat16_2(self, dtype=torch.bfloat16):
        input = torch.randn((512, 30522), requires_grad=True, dtype=dtype)
        target = torch.randint(30522, (512,), dtype=torch.int64)
        loss = F.cross_entropy(input, target)
        #print("cpu result: ", loss)
        loss_xpu = F.cross_entropy(input.xpu(), target.xpu())
        #print("xpu result: ", loss_xpu.cpu())
        self.assertEqual(loss, loss_xpu.cpu())

    def test_cross_entropy_float16_2(self, dtype=torch.bfloat16):
        input = torch.randn((512, 30522), requires_grad=True, dtype=dtype)
        target = torch.randint(30522, (512,), dtype=torch.int64)
        loss = F.cross_entropy(input, target)
        dtype_dpcpp = torch.float16
        #print("cpu result: ", loss)
        loss_xpu = F.cross_entropy(input.xpu().to(dtype_dpcpp), target.xpu())
        #print("xpu result: ", loss_xpu.cpu())
        self.assertEqual(loss, loss_xpu.cpu().to(torch.bfloat16))