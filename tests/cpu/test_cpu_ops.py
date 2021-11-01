import unittest, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import intel_extension_for_pytorch as ipex
from common_utils import TestCase

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

class CPUOPsTester(TestCase):

    def test_channelshuffle(self):
        channel_shuffle = torch.nn.ChannelShuffle(20)
        x = torch.randn(3, 40, 20, 20)
        x1 = x.clone()
        y1 = channel_shuffle(x1)

        # test channels last
        x2 = x.clone().to(memory_format=torch.channels_last)
        y2 = channel_shuffle(x2)
        self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y1, y2)

    def test_pixel_shuffle_unshuffle(self):
        def _test_pixel_shuffle_unshuffle_helper(num_input_dims, valid_channels_dim=True,
                                                 upscale_factor=None):
            # Function to imperatively ensure pixels are shuffled to the correct locations.
            # Used to validate the batch operations in pixel_shuffle.
            def _verify_pixel_shuffle(input, output, upscale_factor):
                for c in range(output.size(-3)):
                    for h in range(output.size(-2)):
                        for w in range(output.size(-1)):
                            height_idx = h // upscale_factor
                            weight_idx = w // upscale_factor
                            channel_idx = (upscale_factor * (h % upscale_factor)) + (w % upscale_factor) + \
                                          (c * upscale_factor ** 2)
                            self.assertEqual(output[..., c, h, w], input[..., channel_idx, height_idx, weight_idx])
            upscale_factor = random.randint(2, 5) if upscale_factor is None else upscale_factor
            # If valid_channels_dim=False, add 1 to make channels dim indivisible by upscale_factor ** 2.
            channels = random.randint(1, 4) * upscale_factor ** 2 + (0 if valid_channels_dim else 1)
            height = random.randint(5, 10)
            width = random.randint(5, 10)
            if num_input_dims == 1:
                input = torch.rand(channels, requires_grad=True)
            elif num_input_dims == 2:
                input = torch.rand(height, width, requires_grad=True)
            else:
                batch_sizes = [random.randint(1, 3) for _ in range(num_input_dims - 3)]
                input = torch.rand(*batch_sizes, channels, height, width, requires_grad=True)
            ps = nn.PixelShuffle(upscale_factor)
            pus = nn.PixelUnshuffle(downscale_factor=upscale_factor)
            if num_input_dims >= 3 and valid_channels_dim and upscale_factor > 0:
                output = ps(input)
                _verify_pixel_shuffle(input, output, upscale_factor)
                output.backward(output.data)
                self.assertEqual(input.data, input.grad.data)
                # Ensure unshuffle properly inverts shuffle.
                unshuffle_output = pus(output)
                self.assertEqual(input, unshuffle_output)
            else:
                self.assertRaises(RuntimeError, lambda: ps(input))

        def _test_pixel_unshuffle_error_case_helper(num_input_dims, valid_height_dim=True, valid_width_dim=True,
                                                    downscale_factor=None):
            downscale_factor = random.randint(2, 5) if downscale_factor is None else downscale_factor
            channels = random.randint(1, 4)
            # If valid_height_dim=False, add 1 to make height dim indivisible by downscale_factor.
            height = random.randint(3, 5) * abs(downscale_factor) + (0 if valid_height_dim else 1)
            # If valid_width_dim=False, add 1 to make width dim indivisible by downscale_factor.
            width = random.randint(3, 5) * abs(downscale_factor) + (0 if valid_width_dim else 1)
            if num_input_dims == 1:
                input = torch.rand(channels, requires_grad=True)
            elif num_input_dims == 2:
                input = torch.rand(height, width, requires_grad=True)
            else:
                batch_sizes = [random.randint(1, 3) for _ in range(num_input_dims - 3)]
                input = torch.rand(*batch_sizes, channels, height, width, requires_grad=True)
            pus = nn.PixelUnshuffle(downscale_factor)
            self.assertRaises(RuntimeError, lambda: pus(input))

        def _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims):
            # For 1D - 2D, this is an error case.
            # For 3D - 5D, this is a success case for pixel_shuffle + pixel_unshuffle.
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims)
            # Error cases for pixel_shuffle.
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, valid_channels_dim=False)
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, upscale_factor=0)
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, upscale_factor=-2)
            # Error cases for pixel_unshuffle.
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, valid_height_dim=False)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, valid_width_dim=False)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, downscale_factor=0)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, downscale_factor=-2)

        def test_pixel_shuffle_unshuffle_1D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=1)

        def test_pixel_shuffle_unshuffle_2D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=2)

        def test_pixel_shuffle_unshuffle_3D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=3)

        def test_pixel_shuffle_unshuffle_4D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=4)

        def test_pixel_shuffle_unshuffle_5D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=5)

        test_pixel_shuffle_unshuffle_1D()
        test_pixel_shuffle_unshuffle_2D()
        test_pixel_shuffle_unshuffle_3D()
        test_pixel_shuffle_unshuffle_4D()
        test_pixel_shuffle_unshuffle_5D()

    def test_pixel_shuffle_nhwc_cpu(self):
        input = torch.randn(3, 18, 4, 4, device='cpu')
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        grad = torch.randn(3, 18, 4, 4, device='cpu')
        ps = torch.nn.PixelShuffle(3)
        pus = torch.nn.PixelUnshuffle(3)

        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_ps = torch.nn.PixelShuffle(3)
        ref_pus = torch.nn.PixelUnshuffle(3)

        out = pus(ps(input))
        out.backward(grad)
        ref_out = ref_pus(ref_ps(ref_input))
        ref_out.backward(ref_grad)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)
        self.assertEqual(input.grad, ref_input.grad)

    def test_batch_norm(self):
        m = nn.BatchNorm2d(100)
        x = torch.randn(20, 100, 35, 45)
        x1 = x.clone().detach().requires_grad_()
        y1 = m(x1)
        y1.mean().backward()

        # test channels last
        x2 = x.clone().detach().to(memory_format=torch.channels_last).requires_grad_()
        y2 = m(x2)
        y2.mean().backward()
        self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y1, y2)
        self.assertTrue(x2.grad.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(x1.grad, x2.grad)

        # test bfloat16
        x3 = x.clone().detach().bfloat16().requires_grad_()
        y3 = m(x3)
        y3.mean().backward()
        self.assertTrue(y3.dtype == torch.bfloat16)
        self.assertEqual(y1, y3, prec=0.1)
        self.assertTrue(x3.grad.dtype == torch.bfloat16)
        self.assertEqual(x1.grad, x3.grad)

        # test autocast
        with torch.cpu.amp.autocast():
            for datatype in (torch.bfloat16, torch.float32):
                x4 = x.clone().detach().to(datatype).requires_grad_()
                y4 = m(x4)
                y4.mean().backward()
                self.assertTrue(y4.dtype == datatype)
                self.assertTrue(x4.grad.dtype == datatype)

    def test_adaptive_avg_pool2d(self):
        m = nn.AdaptiveAvgPool2d((5,7))
        x = torch.randn(3, 64, 8, 9)
        x1 = x.clone().detach().requires_grad_()
        y1 = m(x1)
        y1.mean().backward()

        # test channels last
        x2 = x.clone().detach().to(memory_format=torch.channels_last).requires_grad_()
        y2 = m(x2)
        y2.mean().backward()
        self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y1, y2)
        self.assertTrue(x2.grad.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(x1.grad, x2.grad)

        # test bfloat16
        x3 = x.clone().detach().bfloat16().requires_grad_()
        y3 = m(x3)
        y3.mean().backward()
        self.assertTrue(y3.dtype == torch.bfloat16)
        self.assertEqual(y1, y3, prec=0.01)
        self.assertTrue(x3.grad.dtype == torch.bfloat16)
        self.assertEqual(x1.grad, x3.grad)

        # test autocast
        with torch.cpu.amp.autocast():
            for datatype in (torch.bfloat16, torch.float32):
                x4 = x.clone().detach().to(datatype).requires_grad_()
                y4 = m(x4)
                y4.mean().backward()
                self.assertTrue(y4.dtype == datatype)
                self.assertTrue(x4.grad.dtype == datatype)

    def test_copy(self):
        x = torch.randn(3, 64, 8, 9)
        y = torch.empty(3, 64, 8, 9)
        y.copy_(x)
        self.assertEqual(x, y)

        # test channels last
        y1 = torch.empty(3, 64, 8, 9).to(memory_format=torch.channels_last)
        y1.copy_(x)
        self.assertTrue(y1.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(x, y1)

        # test bfloat16
        y2 = torch.empty(3, 64, 8, 9).bfloat16()
        y2.copy_(x)
        self.assertTrue(y2.dtype == torch.bfloat16)
        self.assertEqual(x, y2, prec=0.01)

    def test_max_pool2d(self):
        m = nn.MaxPool2d((3, 2), stride=(2, 1))
        x = torch.randn(20, 16, 50, 32)
        x1 = x.clone().detach().requires_grad_()
        y1 = m(x1)
        y1.mean().backward()

        # test channels last
        x2 = x.clone().detach().to(memory_format=torch.channels_last).requires_grad_()
        y2 = m(x2)
        y2.mean().backward()
        self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y1, y2)
        self.assertTrue(x2.grad.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(x1.grad, x2.grad)

        # test bfloat16
        x3 = x.clone().detach().bfloat16().requires_grad_()
        y3 = m(x3)
        y3.mean().backward()
        self.assertTrue(y3.dtype == torch.bfloat16)
        self.assertEqual(y1, y3, prec=0.02)
        self.assertTrue(x3.grad.dtype == torch.bfloat16)
        self.assertEqual(x1.grad, x3.grad, prec=1e-4)

        # test autocast
        with torch.cpu.amp.autocast():
            for datatype in (torch.bfloat16, torch.float32):
                x4 = x.clone().detach().to(datatype).requires_grad_()
                y4 = m(x4)
                y4.mean().backward()
                self.assertTrue(y4.dtype == datatype)
                self.assertTrue(x4.grad.dtype == datatype)

    def test_upsample_nearest1d(self):
        x = torch.randn(2, 2, 4)
        x1 = x.clone().detach().requires_grad_()
        y1 = F.interpolate(x1, scale_factor = 2, mode='nearest')
        y1.mean().backward()

        # test bfloat16
        x3 = x.clone().detach().bfloat16().requires_grad_()
        y3 = F.interpolate(x3, scale_factor = 2, mode='nearest')
        y3.mean().backward()
        self.assertTrue(y3.dtype == torch.bfloat16)
        self.assertEqual(y1, y3, prec=0.01)
        self.assertTrue(x3.grad.dtype == torch.bfloat16)
        self.assertEqual(x1.grad, x3.grad)

        # test autocast
        with torch.cpu.amp.autocast():
            for datatype in (torch.bfloat16, torch.float32):
                x4 = x.clone().detach().to(datatype).requires_grad_()
                y4 = F.interpolate(x4, scale_factor = 2, mode='nearest')
                y4.mean().backward()
                self.assertTrue(y4.dtype == datatype)
                self.assertTrue(x4.grad.dtype == datatype)

    def test_upsample_nearest2d(self):
        x = torch.randn(2, 2, 4, 4)
        x1 = x.clone().detach().requires_grad_()
        y1 = F.interpolate(x1, scale_factor = 2, mode='nearest')
        y1.mean().backward()

        # test channels last
        x2 = x.clone().detach().to(memory_format=torch.channels_last).requires_grad_()
        y2 = F.interpolate(x2, scale_factor = 2, mode='nearest')
        y2.mean().backward()
        self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y1, y2)
        self.assertTrue(x2.grad.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(x1.grad, x2.grad)

        # test bfloat16
        x3 = x.clone().detach().bfloat16().requires_grad_()
        y3 = F.interpolate(x3, scale_factor = 2, mode='nearest')
        y3.mean().backward()
        self.assertTrue(y3.dtype == torch.bfloat16)
        self.assertEqual(y1, y3, prec=0.01)
        self.assertTrue(x3.grad.dtype == torch.bfloat16)
        self.assertEqual(x1.grad, x3.grad)

        # test autocast
        with torch.cpu.amp.autocast():
            for datatype in (torch.bfloat16, torch.float32):
                x4 = x.clone().detach().to(datatype).requires_grad_()
                y4 = F.interpolate(x4, scale_factor = 2, mode='nearest')
                y4.mean().backward()
                self.assertTrue(y4.dtype == datatype)
                self.assertTrue(x4.grad.dtype == datatype)

    def test_upsample_nearest3d(self):
        x = torch.randn(2, 2, 2, 4, 4)
        x1 = x.clone().detach().requires_grad_()
        y1 = F.interpolate(x1, scale_factor = 2, mode='nearest')
        y1.mean().backward()

        # test bfloat16
        x3 = x.clone().detach().bfloat16().requires_grad_()
        y3 = F.interpolate(x3, scale_factor = 2, mode='nearest')
        y3.mean().backward()
        self.assertTrue(y3.dtype == torch.bfloat16)
        self.assertEqual(y1, y3, prec=0.01)
        self.assertTrue(x3.grad.dtype == torch.bfloat16)
        self.assertEqual(x1.grad, x3.grad)

        # test autocast
        with torch.cpu.amp.autocast():
            for datatype in (torch.bfloat16, torch.float32):
                x4 = x.clone().detach().to(datatype).requires_grad_()
                y4 = F.interpolate(x4, scale_factor = 2, mode='nearest')
                y4.mean().backward()
                self.assertTrue(y4.dtype == datatype)
                self.assertTrue(x4.grad.dtype == datatype)

    def test_upsample_linear1d(self):
        x = torch.randn(2, 2, 4)
        x1 = x.clone().detach().requires_grad_()
        y1 = F.interpolate(x1, scale_factor = 2, mode='linear')
        y1.mean().backward()

        # test bfloat16
        x3 = x.clone().detach().bfloat16().requires_grad_()
        y3 = F.interpolate(x3, scale_factor = 2, mode='linear')
        y3.mean().backward()
        self.assertTrue(y3.dtype == torch.bfloat16)
        self.assertEqual(y1, y3, prec=0.01)
        self.assertTrue(x3.grad.dtype == torch.bfloat16)
        self.assertEqual(x1.grad, x3.grad)

        # test autocast
        with torch.cpu.amp.autocast():
            for datatype in (torch.bfloat16, torch.float32):
                x4 = x.clone().detach().to(datatype).requires_grad_()
                y4 = F.interpolate(x4, scale_factor = 2, mode='linear')
                y4.mean().backward()
                self.assertTrue(y4.dtype == datatype)
                self.assertTrue(x4.grad.dtype == datatype)

    def test_upsample_bilinear2d(self):
        x = torch.randn(2, 2, 4, 4)
        x1 = x.clone().detach().requires_grad_()
        y1 = F.interpolate(x1, scale_factor = 2, mode='bilinear')
        y1.mean().backward()

        # test channels last
        x2 = x.clone().detach().to(memory_format=torch.channels_last).requires_grad_()
        y2 = F.interpolate(x2, scale_factor = 2, mode='bilinear')
        y2.mean().backward()
        self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y1, y2)
        self.assertTrue(x2.grad.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(x1.grad, x2.grad)

        # test bfloat16
        x3 = x.clone().detach().bfloat16().requires_grad_()
        y3 = F.interpolate(x3, scale_factor = 2, mode='bilinear')
        y3.mean().backward()
        self.assertTrue(y3.dtype == torch.bfloat16)
        self.assertEqual(y1, y3, prec=0.01)
        self.assertTrue(x3.grad.dtype == torch.bfloat16)
        self.assertEqual(x1.grad, x3.grad)

        # test autocast
        with torch.cpu.amp.autocast():
            for datatype in (torch.bfloat16, torch.float32):
                x4 = x.clone().detach().to(datatype).requires_grad_()
                y4 = F.interpolate(x4, scale_factor = 2, mode='bilinear')
                y4.mean().backward()
                self.assertTrue(y4.dtype == datatype)
                self.assertTrue(x4.grad.dtype == datatype)

    def test_upsample_trilinear3d(self):
        x = torch.randn(2, 2, 2, 4, 4)
        x1 = x.clone().detach().requires_grad_()
        y1 = F.interpolate(x1, scale_factor = 2, mode='trilinear')
        y1.mean().backward()

        # test bfloat16
        x3 = x.clone().detach().bfloat16().requires_grad_()
        y3 = F.interpolate(x3, scale_factor = 2, mode='trilinear')
        y3.mean().backward()
        self.assertTrue(y3.dtype == torch.bfloat16)
        self.assertEqual(y1, y3, prec=0.02)
        self.assertTrue(x3.grad.dtype == torch.bfloat16)
        self.assertEqual(x1.grad, x3.grad)

        # test autocast
        with torch.cpu.amp.autocast():
            for datatype in (torch.bfloat16, torch.float32):
                x4 = x.clone().detach().to(datatype).requires_grad_()
                y4 = F.interpolate(x4, scale_factor = 2, mode='trilinear')
                y4.mean().backward()
                self.assertTrue(y4.dtype == datatype)
                self.assertTrue(x4.grad.dtype == datatype)

    def test_avg_pool2d(self):
        m = nn.AvgPool2d((3, 2), stride=(2, 1))
        x = torch.randn(20, 16, 50, 32)
        x1 = x.clone().detach().requires_grad_()
        y1 = m(x1)
        y1.mean().backward()

        # test channels last
        x2 = x.clone().detach().to(memory_format=torch.channels_last).requires_grad_()
        y2 = m(x2)
        y2.mean().backward()
        self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y1, y2)
        self.assertTrue(x2.grad.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(x1.grad, x2.grad)

        # test bfloat16
        x3 = x.clone().detach().bfloat16().requires_grad_()
        y3 = m(x3)
        y3.mean().backward()
        self.assertTrue(y3.dtype == torch.bfloat16)
        self.assertEqual(y1, y3, prec=0.01)
        self.assertTrue(x3.grad.dtype == torch.bfloat16)
        self.assertEqual(x1.grad, x3.grad)

        # test autocast
        with torch.cpu.amp.autocast():
            for datatype in (torch.bfloat16, torch.float32):
                x4 = x.clone().detach().to(datatype).requires_grad_()
                y4 = m(x4)
                y4.mean().backward()
                self.assertTrue(y4.dtype == datatype)
                self.assertTrue(x4.grad.dtype == datatype)

    def test_adaptive_max_pool2d(self):
        m = nn.AdaptiveMaxPool2d((5,7))
        x = torch.randn(3, 64, 8, 9)
        x1 = x.clone().detach().requires_grad_()
        y1 = m(x1)
        y1.mean().backward()

        # test channels last
        x2 = x.clone().detach().to(memory_format=torch.channels_last).requires_grad_()
        y2 = m(x2)
        y2.mean().backward()
        self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(y1, y2)
        self.assertTrue(x2.grad.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(x1.grad, x2.grad)

        # test bfloat16
        x3 = x.clone().detach().bfloat16().requires_grad_()
        y3 = m(x3)
        y3.mean().backward()
        self.assertTrue(y3.dtype == torch.bfloat16)
        self.assertEqual(y1, y3, prec=0.01)
        self.assertTrue(x3.grad.dtype == torch.bfloat16)
        self.assertEqual(x1.grad, x3.grad, prec=0.001)

        # test autocast
        with torch.cpu.amp.autocast():
            for datatype in (torch.bfloat16, torch.float32):
                x4 = x.clone().detach().to(datatype).requires_grad_()
                y4 = m(x4)
                y4.mean().backward()
                self.assertTrue(y4.dtype == datatype)
                self.assertTrue(x4.grad.dtype == datatype)

    @skipIfNoTorchVision
    def test_torchvision_nms(self):
        num_boxes = 50
        boxes = torch.rand(num_boxes, 4)
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.randn(num_boxes)
        y1 = torchvision.ops.nms(boxes, scores, 0.5)
        with torch.cpu.amp.autocast():
            y2 = torchvision.ops.nms(boxes.bfloat16(), scores.bfloat16(), 0.5)
            self.assertEqual(y1, y2)


if __name__ == '__main__':
    test = unittest.main()
