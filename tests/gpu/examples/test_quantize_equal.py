import torch
from torch.testing._internal.common_utils import TestCase

import ipex

import pytest


class TestTorchMethod(TestCase):
    def test_q_equal(self, dtype=torch.float):
        zero_point = 0
        scale = 0.4
        dtype = torch.quint8

        a = torch.randn(1, 2, 5, 5)
        b = torch.randn(2, 2, 5, 5)

        q_a = torch.quantize_per_tensor(a, scale, zero_point, dtype)
        q_b = torch.quantize_per_tensor(b, scale, zero_point, dtype)

        self.assertEqual(q_a.equal(q_b), 0)
        self.assertEqual(q_a.equal(q_a), 1)
