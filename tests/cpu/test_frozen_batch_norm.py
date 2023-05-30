# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import unittest
from common_utils import TestCase
from intel_extension_for_pytorch.nn import FrozenBatchNorm2d

try:
    import torchvision  # noqa: F401
    from torchvision.ops.misc import FrozenBatchNorm2d as FrozenBN2d

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


class FrozenBNTester(TestCase):
    @skipIfNoTorchVision
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
        x2 = (
            input.clone()
            .detach()
            .to(memory_format=torch.channels_last)
            .requires_grad_()
        )
        y2 = m(x2)
        self.assertTrue(y2.dtype == torch.float32)
        self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y2, y1)

        y2.mean().backward()
        self.assertTrue(x2.grad.dtype == torch.float32)
        self.assertTrue(x2.grad.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(x2.grad, x1.grad)

    @skipIfNoTorchVision
    def test_frozen_batch_norm_bfloat16(self):
        m = FrozenBatchNorm2d(100)
        m1 = FrozenBN2d(100)
        running_mean = torch.randn(100)
        running_var = torch.randn(100)
        m.running_mean = running_mean
        m.running_var = running_var
        m1.running_mean = running_mean
        m1.running_var = running_var
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
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
            x2 = (
                input.clone()
                .detach()
                .to(memory_format=torch.channels_last)
                .requires_grad_()
            )
            y2 = m(x2)
            self.assertTrue(y2.dtype == torch.bfloat16)
            self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(y2, y1, prec=0.1)

            y2.mean().backward()
            self.assertTrue(x2.grad.dtype == torch.bfloat16)
            self.assertTrue(x2.grad.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(x2.grad, x1.grad)


if __name__ == "__main__":
    test = unittest.main()
