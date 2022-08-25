import unittest, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import intel_extension_for_pytorch as ipex
from common_utils import TestCase
import itertools

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

bn_m = {1 : nn.BatchNorm1d, 2 : nn.BatchNorm2d, 3 : nn.BatchNorm3d}

class CPUOPsTester(TestCase):

    # Keep this UT temporarily to make sure the OP behavior in PyTorch is as expected.
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

    # Keep this UT temporarily to make sure the OP behavior in PyTorch is as expected.
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

    # Keep this UT temporarily to make sure the OP behavior in PyTorch is as expected.
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
        for dim in [2, 3]:
            m = bn_m[dim](10)
            input_size = [3, 10, 25, 25]
            if dim == 3:
                input_size.append(25)
            x = torch.randn(input_size)
            x1 = x.clone().detach().requires_grad_()
            y1 = m(x1)
            y1.mean().backward()

            # test channels last
            suggest_memory_format = torch.channels_last if dim == 2 else torch.channels_last_3d
            x2 = x.clone().detach().to(memory_format=suggest_memory_format).requires_grad_()

            y2 = m(x2)
            y2.mean().backward()
            self.assertTrue(y2.is_contiguous(memory_format=suggest_memory_format))
            self.assertEqual(y1, y2)
            self.assertTrue(x2.grad.is_contiguous(memory_format=suggest_memory_format))
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

                    x5 = x.clone().detach().to(datatype).to(memory_format=suggest_memory_format).requires_grad_()
                    y5 = m(x5)
                    y5.mean().backward()
                    self.assertTrue(y5.dtype == datatype)
                    self.assertTrue(x5.grad.dtype == datatype)
                    self.assertTrue(y5.is_contiguous(memory_format=suggest_memory_format))
                    self.assertTrue(x5.grad.is_contiguous(memory_format=suggest_memory_format))

            # test non-contiguous inputs
            x6 = torch.transpose(x.clone().detach(), 2, 3).requires_grad_()
            x_ref = x6.clone().detach().contiguous().requires_grad_()
            y6 = m(x6)
            y6.mean().backward()
            y_ref = m(x_ref)
            y_ref.mean().backward()
            self.assertEqual(y6, y_ref)
            self.assertEqual(x6.grad, x_ref.grad) 

    # Keep this UT temporarily to make sure the OP behavior in PyTorch is as expected.
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

                x5 = x.clone().detach().to(datatype).to(memory_format=torch.channels_last).requires_grad_()
                y5 = m(x5)
                y5.mean().backward()
                self.assertTrue(y5.dtype == datatype)
                self.assertTrue(x5.grad.dtype == datatype)
                self.assertTrue(y5.is_contiguous(memory_format=torch.channels_last))
                self.assertTrue(x5.grad.is_contiguous(memory_format=torch.channels_last))

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

    # Keep this UT temporarily to make sure the OP behavior in PyTorch is as expected.
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

                x5 = x.clone().detach().to(datatype).to(memory_format=torch.channels_last).requires_grad_()
                y5 = m(x5)
                y5.mean().backward()
                self.assertTrue(y5.dtype == datatype)
                self.assertTrue(x5.grad.dtype == datatype)
                self.assertTrue(y5.is_contiguous(memory_format=torch.channels_last))
                self.assertTrue(x5.grad.is_contiguous(memory_format=torch.channels_last))

    # Keep this UT temporarily to make sure the OP behavior in PyTorch is as expected.
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

    # Keep this UT temporarily to make sure the OP behavior in PyTorch is as expected.
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

                x5 = x.clone().detach().to(datatype).to(memory_format=torch.channels_last).requires_grad_()
                y5 = F.interpolate(x5, scale_factor = 2, mode='nearest')
                y5.mean().backward()
                self.assertTrue(y5.dtype == datatype)
                self.assertTrue(x5.grad.dtype == datatype)
                self.assertTrue(y5.is_contiguous(memory_format=torch.channels_last))
                self.assertTrue(x5.grad.is_contiguous(memory_format=torch.channels_last))

    # Keep this UT temporarily to make sure the OP behavior in PyTorch is as expected.
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

                x5 = x.clone().detach().to(datatype).to(memory_format=torch.channels_last_3d).requires_grad_()
                y5 = F.interpolate(x5, scale_factor = 2, mode='nearest')
                y5.mean().backward()
                self.assertTrue(y5.dtype == datatype)
                self.assertTrue(x5.grad.dtype == datatype)
                self.assertTrue(y5.is_contiguous(memory_format=torch.channels_last_3d))
                self.assertTrue(x5.grad.is_contiguous(memory_format=torch.channels_last_3d))

    # Keep this UT temporarily to make sure the OP behavior in PyTorch is as expected.
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

    # Keep this UT temporarily to make sure the OP behavior in PyTorch is as expected.
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

                x5 = x.clone().detach().to(datatype).to(memory_format=torch.channels_last).requires_grad_()
                y5 = F.interpolate(x5, scale_factor = 2, mode='bilinear')
                y5.mean().backward()
                self.assertTrue(y5.dtype == datatype)
                self.assertTrue(x5.grad.dtype == datatype)
                self.assertTrue(y5.is_contiguous(memory_format=torch.channels_last))
                self.assertTrue(x5.grad.is_contiguous(memory_format=torch.channels_last))

    # Keep this UT temporarily to make sure the OP behavior in PyTorch is as expected.
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

                x5 = x.clone().detach().to(datatype).to(memory_format=torch.channels_last_3d).requires_grad_()
                y5 = F.interpolate(x5, scale_factor = 2, mode='trilinear')
                y5.mean().backward()
                self.assertTrue(y5.dtype == datatype)
                self.assertTrue(x5.grad.dtype == datatype)
                self.assertTrue(y5.is_contiguous(memory_format=torch.channels_last_3d))
                self.assertTrue(x5.grad.is_contiguous(memory_format=torch.channels_last_3d))

    def test_groupnorm_nhwc(self):
        def helper(self, size, groups, memory_format, dtype, prec=1e-5):
            channels = size[1]
            input = torch.randn(size, dtype=dtype, requires_grad=True)
            input = input.contiguous(memory_format=memory_format)
            input.retain_grad()
            grad = torch.randn(size, dtype=dtype)
            grad = grad.contiguous(memory_format=memory_format)
            gn = nn.GroupNorm(groups, channels).to(dtype)
            gn.weight.data.uniform_()
            gn.bias.data.uniform_()

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_gn = nn.GroupNorm(groups, channels).to(dtype)
            ref_gn.load_state_dict(gn.state_dict())

            out = gn(input)
            out.backward(grad)
            ref_out = ref_gn(ref_input)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=memory_format))
            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(out.dtype == dtype)
            self.assertTrue(input.grad.dtype == dtype)
            self.assertEqual(out, ref_out, prec = prec)
            self.assertEqual(input.grad, ref_input.grad, prec = prec)
            if (dtype == torch.float32):
                self.assertEqual(gn.weight.grad, ref_gn.weight.grad, prec=1e-04)
                self.assertEqual(gn.bias.grad, ref_gn.bias.grad, prec=1e-04)
        helper(self, (4, 8, 10, 10), 4, torch.channels_last, torch.float32)
        helper(self, (2, 30, 9, 9), 3, torch.channels_last, torch.float32)
        helper(self, (2, 9, 7, 11, 15), 3, torch.channels_last_3d, torch.float32)
        helper(self, (4, 8, 10, 10), 4, torch.channels_last, torch.bfloat16, prec=0.04)
        helper(self, (2, 30, 9, 9), 3, torch.channels_last, torch.bfloat16, prec=0.04)
        helper(self, (2, 9, 7, 11, 15), 3, torch.channels_last_3d, torch.bfloat16, prec=0.04)

    def test_groupnorm_nwc(self):
        size = (4, 20, 20)
        channels = size[1]
        groups = 4
        x = torch.randn(size, requires_grad=True)
        grad = torch.randn(size)
        m = nn.GroupNorm(groups, channels)

        # test nwc
        x1 = x.clone().detach().requires_grad_().transpose(1, 2)
        grad1 = grad.detach().clone()
        y1 = m(x1)
        y1.backward(grad1)

        x2 = x1.clone().detach().contiguous()
        grad2 = grad.detach().clone()
        y2 = m(x2)
        y2.backward(grad2)
        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad)

        # test bfloat16
        x3 = x.clone().detach().requires_grad_().transpose(1, 2).bfloat16()
        grad3 = grad.detach().clone()
        m_bf16 = m.to(torch.bfloat16)
        y3 = m_bf16(x3)
        self.assertTrue(y3.dtype == torch.bfloat16)
        self.assertEqual(y3, y2, prec=0.02)
        self.assertEqual(x3.grad, x2.grad, prec=0.02)
        self.assertEqual(m.weight.grad, m_bf16.weight.grad)
        self.assertEqual(m.bias.grad, m_bf16.bias.grad)

    def test_avg_pool2d(self):
        models = [nn.AvgPool2d((3, 2), stride=(2, 1)), nn.AvgPool2d((2, 2), stride=(1, 4), ceil_mode=True, padding=(-2, -2))]
        inputs = [torch.randn(20, 16, 50, 32), torch.randn(3, 16, 16, 8)]
        for i in range(len(models)):
            m = models[i]
            x = inputs[i]
            x1 = x.clone().detach().requires_grad_()
            y1 = m(x1)
            y1.backward(y1.data)

            # test channels last
            x2 = x.clone().detach().to(memory_format=torch.channels_last).requires_grad_()
            y2 = m(x2)
            y2.backward(y2.data)
            self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(y1, y2)
            self.assertTrue(x2.grad.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(x1.grad, x2.grad)

            # test bfloat16
            x3 = x.clone().detach().bfloat16().requires_grad_()
            y3 = m(x3)
            y3.backward(y3.data)
            self.assertTrue(y3.dtype == torch.bfloat16)
            self.assertEqual(y1, y3, prec=0.01)
            self.assertTrue(x3.grad.dtype == torch.bfloat16)
            self.assertEqual(x1.grad, x3.grad, prec=0.01)

            # test autocast
            with torch.cpu.amp.autocast():
                for datatype in (torch.bfloat16, torch.float32):
                    x4 = x.clone().detach().to(datatype).requires_grad_()
                    y4 = m(x4)
                    y4.backward(y4.data)
                    self.assertTrue(y4.dtype == datatype)
                    self.assertTrue(x4.grad.dtype == datatype)

                    x5 = x.clone().detach().to(datatype).to(memory_format=torch.channels_last).requires_grad_()
                    y5 = m(x5)
                    y5.backward(y5.data)
                    self.assertTrue(y5.dtype == datatype)
                    self.assertTrue(x5.grad.dtype == datatype)
                    self.assertTrue(y5.is_contiguous(memory_format=torch.channels_last))
                    self.assertTrue(x5.grad.is_contiguous(memory_format=torch.channels_last))

    # Keep this UT temporarily to make sure the OP behavior in PyTorch is as expected.
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

                x5 = x.clone().detach().to(datatype).to(memory_format=torch.channels_last).requires_grad_()
                y5 = m(x5)
                y5.mean().backward()
                self.assertTrue(y5.dtype == datatype)
                self.assertTrue(x5.grad.dtype == datatype)
                self.assertTrue(y5.is_contiguous(memory_format=torch.channels_last))
                self.assertTrue(x5.grad.is_contiguous(memory_format=torch.channels_last))

    def test_avg_pool3d_ndhwc(self):
        def helper(n, c, d, h, w, kernel_size, dtype, contig,
                   count_include_pad=True, divisor_override=None):
            input = torch.randint(1, 10, (n, c, d, h, w), device='cpu', dtype=dtype)
            input = input.contiguous(memory_format=torch.channels_last_3d)
            if not contig:
                input = input[:, ::2, :, :, :]
            input.requires_grad_(True)
            pool = torch.nn.AvgPool3d(kernel_size=kernel_size, count_include_pad=count_include_pad,
                                      divisor_override=divisor_override)

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_pool = torch.nn.AvgPool3d(kernel_size=kernel_size, count_include_pad=count_include_pad,
                                          divisor_override=divisor_override)

            out = pool(input)
            out.backward(out.data)
            ref_out = ref_pool(ref_input)
            ref_out.backward(ref_out.data)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last_3d))
            self.assertTrue(ref_out.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(input.grad, ref_input.grad)

        for dtype in [torch.float32, torch.double]:
            for contig in [True, False]:
                for count_include_pad in [True, False]:
                    helper(4, 8, 10, 10, 10, (3, 2, 3), dtype, contig, count_include_pad=count_include_pad)
                    helper(4, 8, 18, 9, 14, (2, 3, 2), dtype, contig, count_include_pad=count_include_pad)
                    helper(4, 8, 7, 8, 9, (2, 2, 2), dtype, contig,
                           count_include_pad=count_include_pad, divisor_override=100)

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

    def test_mean(self):
        x = torch.randn(1, 64, 100, 13, 24, requires_grad=True)
        for dtype in [torch.float32, torch.double, torch.bfloat16]:
            y1 = torch.mean(x, dim=(3, 4), keepdim=False, dtype=dtype)
            x2 = x.clone().detach().to(memory_format=torch.channels_last_3d).requires_grad_()
            y2 = torch.mean(x2, dim=(3, 4), keepdim=False, dtype=dtype)
            self.assertEqual(y1, y2)

    def test_sum(self):
        dtypes = [torch.float32, torch.double, torch.bfloat16, torch.float16]

        x1 = torch.randn((1, 128, 56, 56)).to(memory_format=torch.channels_last)
        x1 = x1.reshape([1, 2, 64, 56, 56])
        x2 = x1.contiguous()
        for dtype in dtypes:
            y1 = torch.sum(x1, dim=(1), keepdim=False, dtype=dtype)
            y2 = torch.sum(x2, dim=(1), keepdim=False, dtype=dtype)
            self.assertEqual(y1, y2, prec=1e-4)

        x3 = torch.randn((1, 64, 100, 13, 24)).to(memory_format=torch.channels_last_3d)
        x4 = x3.contiguous()
        for dtype in dtypes:
            y3 = torch.sum(x3, dim=(3, 4), keepdim=False, dtype=dtype)
            y4 = torch.sum(x4, dim=(3, 4), keepdim=False, dtype=dtype)
            self.assertEqual(y3, y4, prec=1e-4)

        a = torch.randn([3, 2, 3])
        mask = a.ge(0.5)
        s = mask.sum()
        self.assertTrue(s.dtype != torch.bool)

        # add ut for special case - not a true reduction in sumkernel
        x5 = torch.rand(789, 357)
        x6 = x5.transpose(0, 1)
        y5 = torch.mvlgamma(x5, p=1)
        y6 = torch.mvlgamma(x6, p=1).transpose(0, 1)
        self.assertEqual(y5, y6)
    
    def test_matmul(self):
        dtypes = [torch.float32, torch.bfloat16]
        for dtype in dtypes:
            a = torch.randn(2, 3, dtype=dtype)
            b = torch.randn(3, 4, dtype=dtype)
            c = torch.zeros(2, 4, dtype=dtype)
            torch.mm(a, b, out=c)
            d = torch.mm(a, b)
            self.assertTrue(torch.equal(c, d))
            e = torch.randn(10, 3, 4, dtype=dtype)
            f = torch.randn(10, 4, 5, dtype=dtype)
            g = torch.zeros(10, 3, 5, dtype=dtype)
            torch.bmm(e, f, out=g)
            h = torch.bmm(e, f)
            self.assertTrue(torch.equal(g, h))

    def test_index_select(self):
        x = torch.randn(3, 64, 8, 9)
        indices = torch.tensor([0, 2])
        y = x.index_select(1, indices)

        # test bfloat16
        x2 = x.clone().detach().bfloat16()
        y2 = x2.index_select(1, indices)
        self.assertTrue(y2.dtype == torch.bfloat16)
        self.assertEqual(y2, y, prec=0.01)

    def test_cat(self):
        for datatype in [torch.float32, torch.double, torch.bfloat16]:
            for dim, size in itertools.product([0, 1], [[2, 1], [2, 2], [5, 10]]):
                x = torch.randn(size, dtype=datatype)
                y = torch.cat([x, x], dim)
                self.assertTrue(y.dtype == datatype)

            # long input tensor list
            x1 = torch.randn((2, 2), dtype=datatype)
            input1 = []
            for i in range(100):
                input1.append(x1)
            y1 = torch.cat(input1, 0)
            self.assertTrue(y1.size() == torch.Size([200, 2]))
            self.assertTrue(y1.dtype == datatype)

            # input tensors have different shapes and strides
            x2 = torch.randn((400, 2), dtype=datatype)
            input2 = []
            for i in range(10):
                input2.append(x1)
            for i in range(100):
                input2.append(x2)
            y2 = torch.cat(input2, 0)
            self.assertTrue(y2.size() == torch.Size([40020, 2]))
            self.assertTrue(y2.dtype == datatype)

            x3 = torch.randn((4000, 2), dtype=datatype)
            input3 = []
            for i in range(10):
                input3.append(x1)
            for i in range(10):
                input3.append(x3)
            y3 = torch.cat(input3, 0)
            self.assertTrue(y3.size() == torch.Size([40020, 2]))
            self.assertTrue(y3.dtype == datatype)

            x4 = torch.randn((4, 2), dtype=datatype)
            input4 = []
            for i in range(10):
                input4.append(x1)
            for i in range(10):
                input4.append(x4)
            y4 = torch.cat(input4, 0)
            self.assertTrue(y4.size() == torch.Size([60, 2]))
            self.assertTrue(y4.dtype == datatype)

            # "out" arg is used but  un-defined
            y5 = torch.cat([x4, x4], 0, out=torch.empty(0, dtype=datatype))
            self.assertEqual(y5, torch.cat([x4, x4], 0))
            self.assertTrue(y5.dtype == datatype)

            # out is defined with wrong shape
            ref = torch.cat([x4, x4], 0)
            out = torch.zeros(1)
            out_ptr = out.data_ptr()
            torch.cat([x4, x4], 0, out=out)
            self.assertEqual(ref, out)
            self.assertTrue(ref.dtype == datatype)
            self.assertTrue(out_ptr != out.data_ptr())

            # out is defined with correct shape
            ref = torch.cat([x4, x4], 0)
            out = torch.zeros_like(ref)
            out_ptr = out.data_ptr()
            torch.cat([x4, x4], 0, out=out)
            self.assertEqual(ref, out)
            self.assertTrue(ref.dtype == datatype)
            self.assertTrue(out_ptr == out.data_ptr())

            y6 = torch.cat([x4, x4], 0, out=torch.empty(0, dtype=torch.float32))
            self.assertEqual(y6, torch.cat([x4, x4], 0))
            self.assertTrue(y6.dtype == torch.float32)

            # one of input tensors is empty
            x7 = torch.empty(0, dtype=datatype)
            y7 = torch.cat([x4, x4, x7], 0)
            self.assertTrue(y7.size() == torch.Size([8, 2]))
            self.assertTrue(y7.dtype == datatype)

if __name__ == '__main__':
    test = unittest.main()
