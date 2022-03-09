import math

import torch
import torch.nn as nn
from torch.distributions import Cauchy
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import pytest

cpu_device = torch.device("cpu")
sycl_device = torch.device("xpu")


def is_all_nan(tensor):
    """
    Checks if all entries of a tensor is nan.
    """
    return (tensor != tensor).all()


class TestTorchMethod(TestCase):
    def test_cauchy(self):
        loc = torch.zeros(5, 5, requires_grad=True, device=sycl_device)
        scale = torch.ones(5, 5, requires_grad=True, device=sycl_device)

        loc_1d = torch.zeros(1, requires_grad=True, device=sycl_device)
        scale_1d = torch.ones(1, requires_grad=True, device=sycl_device)

        eps = loc.new(loc.size()).cauchy_()
        c = Cauchy(loc, scale).rsample()
        c.backward(torch.ones_like(c))

        self.assertTrue(is_all_nan(Cauchy(loc_1d, scale_1d).mean))
        self.assertEqual(Cauchy(loc_1d, scale_1d).variance, math.inf)
        self.assertEqual(Cauchy(loc, scale).sample().size(), (5, 5))
        self.assertEqual(Cauchy(loc, scale).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Cauchy(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(Cauchy(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
