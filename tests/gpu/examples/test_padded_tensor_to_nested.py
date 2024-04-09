import torch
import intel_extension_for_pytorch as ipex  # noqa

from torch.testing._internal.common_utils import TestCase
import pytest  # noqa

xpu_device = "xpu"
cpu_device = "cpu"


class TestTorchMethod(TestCase):
    def test_nested_tensor_from_padded(self, device=cpu_device):
        nested_size = torch.tensor([[1, 2], [2, 2]])
        padded_tensor = torch.randn(2, 2, 2, dtype=torch.float, device=device)
        padded_tensor[0, 1, :] = 0
        padded_tensor.requires_grad_()

        def grad_test_func(tensor, nested_size):
            nt = torch._nested_from_padded(
                tensor, nested_size, fuse_transform_0213=False
            )
            # This implicitly tests to_padded_tensor grads
            return nt

        data = (padded_tensor, nested_size)
        cpu_res = grad_test_func(padded_tensor, nested_size)

        padded_tensor_xpu = padded_tensor.to(xpu_device)
        xpu_res = grad_test_func(padded_tensor_xpu, nested_size)

        print(cpu_res)
        print(xpu_res)
        self.assertEqual(cpu_res, xpu_res.to(cpu_device))
