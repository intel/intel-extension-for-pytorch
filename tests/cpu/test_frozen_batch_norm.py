# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import unittest, copy
from common_utils import TestCase
from intel_extension_for_pytorch.nn import FrozenBatchNorm2d

class FrozenBN2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBN2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

class FrozenBNTester(TestCase):

    def test_frozen_batch_norm(self):
        m = FrozenBatchNorm2d(100)
        m1 = FrozenBN2d(100)
        running_mean = torch.randn(100)
        running_var = torch.randn(100)
        m.running_mean = running_mean
        m.running_var = running_var
        m1.running_mean = running_mean
        m1.running_var = running_var
        input = torch.randn(20, 100, 35, 45)
        x = input.clone().detach().requires_grad_()
        x1 = input.clone().detach().requires_grad_()
        y = m(x)
        y1 = m1(x1)
        self.assertTrue(y.dtype == torch.float32)
        self.assertEqual(y, y1)

        # backward
        y.mean().backward()
        y1.mean().backward()
        self.assertTrue(x.grad.dtype == torch.float32)
        self.assertEqual(x.grad, x1.grad)

        # test channels last
        x2 = input.clone().detach().to(memory_format=torch.channels_last).requires_grad_()
        y2 = m(x2)
        self.assertTrue(y2.dtype == torch.float32)
        self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y2, y1)

        y2.mean().backward()
        self.assertTrue(x2.grad.dtype == torch.float32)
        self.assertTrue(x2.grad.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(x2.grad, x1.grad)

    def test_frozen_batch_norm_bfloat16(self):
        m = FrozenBatchNorm2d(100)
        m1 = FrozenBN2d(100)
        running_mean = torch.randn(100)
        running_var = torch.randn(100)
        m.running_mean = running_mean
        m.running_var = running_var
        m1.running_mean = running_mean
        m1.running_var = running_var
        input = torch.randn(20, 100, 35, 45).bfloat16()
        x = input.clone().detach().requires_grad_()
        x1 = input.clone().detach().requires_grad_()
        y = m(x)
        y1 = m1(x1)
        self.assertTrue(y.dtype == torch.bfloat16)
        self.assertEqual(y, y1, prec=0.1)

        # backward
        y.mean().backward()
        y1.mean().backward()
        self.assertTrue(x.grad.dtype == torch.bfloat16)
        self.assertEqual(x.grad, x1.grad)

        # test channels last
        x2 = input.clone().detach().to(memory_format=torch.channels_last).requires_grad_()
        y2 = m(x2)
        self.assertTrue(y2.dtype == torch.bfloat16)
        self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y2, y1, prec=0.1)

        y2.mean().backward()
        self.assertTrue(x2.grad.dtype == torch.bfloat16)
        self.assertTrue(x2.grad.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(x2.grad, x1.grad)


if __name__ == '__main__':
    test = unittest.main()
