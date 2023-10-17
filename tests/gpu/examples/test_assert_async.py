import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa


class TestTorchMethod(TestCase):
    def test_assert_async(self):
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with no values is ambiguous"):
            torch._assert_async(torch.tensor([]).to("xpu"))
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with more than one value is ambiguous"):
            torch._assert_async(torch.tensor([0, 0]).to("xpu"))
        # with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
        #     torch._assert_async(torch.tensor(0).to("xpu"))
        # with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
        #     torch._assert_async(torch.tensor(0.0).to("xpu"))
        # with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
        #     torch._assert_async(torch.tensor(False).to("xpu"))
        # with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
        #     torch._assert_async(torch.tensor(0 + 0j).to("xpu"))
        torch._assert_async(torch.tensor(1).to("xpu"))
        torch._assert_async(torch.tensor(0.1).to("xpu"))
        torch._assert_async(torch.tensor(-0.1).to("xpu"))
        torch._assert_async(torch.tensor(True).to("xpu"))
        torch._assert_async(torch.tensor(0 + 0.1j).to("xpu"))

    def test_assert_async_msg(self):
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with no values is ambiguous"):
            torch._assert_async(torch.tensor([]).to("xpu"), 'test')
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with more than one value is ambiguous"):
            torch._assert_async(torch.tensor([0, 0]).to("xpu"), 'test')
        # with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
        #     torch._assert_async(torch.tensor(0).to("xpu"), 'test')
        # with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
        #     torch._assert_async(torch.tensor(0.0).to("xpu"), 'test')
        # with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
        #     torch._assert_async(torch.tensor(False).to("xpu"), 'test')
        # with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
        #     torch._assert_async(torch.tensor(0 + 0j).to("xpu"), 'test')
        torch._assert_async(torch.tensor(1).to("xpu"), 'test')
        torch._assert_async(torch.tensor(0.1).to("xpu"), 'test')
        torch._assert_async(torch.tensor(-0.1).to("xpu"), 'test')
        torch._assert_async(torch.tensor(True).to("xpu"), 'test')
        torch._assert_async(torch.tensor(0 + 0.1j).to("xpu"), 'test')
