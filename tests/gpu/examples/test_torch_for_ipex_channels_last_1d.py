import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
import random

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

# Note:
# In order to press the gradient of weight below 1,
# the default weight should be set to 1e-ks (ks is kernel_size).
# For now, precision could not be pressed to 1e-5,
# but only if there is a real model which suffers the accuracy problem,
# we won't delve into this issue.


class TestNNMethod(TestCase):
    def test_memory_format_consistency(self):
        x = torch.randn(10, 3, 1, 1, device=dpcpp_device)
        x_rep = x.as_strided(x.size(), x.stride())
        self.assertEqual(x.size(), x_rep.size())
        self.assertEqual(x.stride(), x_rep.stride())
        self.assertEqual(x.is_contiguous(), x_rep.is_contiguous())

    def _test_memory_format_transformations(
        self,
        input_generator_fn,
        transformation_fn,
        compare_data=True,
        default_is_preserve=False,
    ):
        # xc is a channels last 1d tensor
        xc = input_generator_fn()
        # xc is not memory dense, but looks like channels last 1d
        xc = xc[..., ::2]

        xc = input_generator_fn()
        clone = transformation_fn(xc, memory_format=torch.contiguous_format)
        self.assertTrue(clone.is_contiguous())
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        xc = input_generator_fn()
        clone = transformation_fn(xc)

        if default_is_preserve:
            self.assertFalse(clone.is_contiguous())
        else:
            self.assertTrue(clone.is_contiguous())
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        x = torch.randn((3, 4, 5, 6, 7, 8, 9), device=dpcpp_device)
        for _ in range(10):
            permutation = list(range(len(x.shape)))
            random.shuffle(permutation)
            x = x.permute(permutation)
            self.assertEqual(
                x.stride(),
                transformation_fn(x, memory_format=torch.preserve_format).stride(),
            )
