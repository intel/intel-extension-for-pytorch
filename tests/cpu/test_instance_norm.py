# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

# import unittest
from common_utils import TestCase
from torch.nn import InstanceNorm2d, InstanceNorm3d, BatchNorm2d, BatchNorm3d

bn_m = {2: BatchNorm2d, 3: BatchNorm3d}
inst_m = {2: InstanceNorm2d, 3: InstanceNorm3d}


class InstanceNormTester(TestCase):
    def test_instance_norm(self):
        for dim in [2, 3]:
            batch = 10
            channel = 100

            input_size = [batch, channel]
            bn_size = [1, batch * channel]

            if dim == 2:
                memory_format = torch.channels_last
            else:
                memory_format = torch.channels_last_3d

            if dim == 2:
                input_size += [45, 35]
                bn_size += [45, 35]
            if dim == 3:
                input_size += [45, 35, 100]
                bn_size += [45, 35, 100]

            input = torch.randn(input_size)
            x = input.clone().detach().requires_grad_()
            x1 = input.clone().detach().requires_grad_()
            x1r = x1.reshape(bn_size)

            m = inst_m[dim](channel, affine=True)
            m1 = bn_m[dim](batch * channel, affine=True)

            y = m(x)
            y1 = m1(x1r).reshape_as(x1)
            self.assertTrue(y.dtype == torch.float32)
            self.assertEqual(y, y1)

            # backward
            y.mean().backward()
            y1.mean().backward()
            self.assertTrue(x.grad.dtype == torch.float32)
            self.assertEqual(x.grad, x1.grad)

            # test channels last
            x2 = input.clone().detach().to(memory_format=memory_format).requires_grad_()
            y2 = m(x2)
            self.assertTrue(y2.dtype == torch.float32)
            self.assertEqual(y2, y1)
            self.assertTrue(y2.is_contiguous(memory_format=torch.contiguous_format))

            y2.mean().backward()
            self.assertTrue(x2.grad.dtype == torch.float32)
            self.assertEqual(x2.grad, x1.grad)
            self.assertTrue(x2.grad.is_contiguous(memory_format=memory_format))

    def test_instance_norm_bfloat16(self):
        for dim in [2, 3]:
            batch = 10
            channel = 100

            input_size = [batch, channel]
            bn_size = [1, batch * channel]

            if dim == 2:
                memory_format = torch.channels_last
            else:
                memory_format = torch.channels_last_3d

            if dim == 2:
                input_size += [45, 35]
                bn_size += [45, 35]
            if dim == 3:
                input_size += [45, 35, 100]
                bn_size += [45, 35, 100]

            m = inst_m[dim](channel, affine=True)
            m1 = bn_m[dim](batch * channel, affine=True)

            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                input = torch.randn(input_size).bfloat16()
                x = input.clone().detach().requires_grad_()
                x1 = input.clone().detach().requires_grad_()
                x1r = x1.reshape(bn_size)

                y = m(x)
                y1 = m1(x1r).reshape_as(x1)
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
                    .to(memory_format=memory_format)
                    .requires_grad_()
                )
                y2 = m(x2)
                self.assertTrue(y2.dtype == torch.bfloat16)
                self.assertTrue(y2.is_contiguous(memory_format=torch.contiguous_format))
                self.assertEqual(y2, y1, prec=0.1)

                y2.mean().backward()
                self.assertTrue(x2.grad.dtype == torch.bfloat16)
                self.assertTrue(x2.grad.is_contiguous(memory_format=memory_format))
                self.assertEqual(x2.grad, x1.grad)


# if __name__ == "__main__":
#     test = unittest.main()
