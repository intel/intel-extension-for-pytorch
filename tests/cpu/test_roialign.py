import unittest, copy
import torch
import intel_extension_for_pytorch as ipex
from common_utils import TestCase

import numpy as np
import math

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

def bilinear_interpolate(data, y, x, snap_border=False):
    height, width = data.shape

    if snap_border:
        if -1 < y <= 0:
            y = 0
        elif height - 1 <= y < height:
            y = height - 1

        if -1 < x <= 0:
            x = 0
        elif width - 1 <= x < width:
            x = width - 1

    y_low = int(math.floor(y))
    x_low = int(math.floor(x))
    y_high = y_low + 1
    x_high = x_low + 1

    wy_h = y - y_low
    wx_h = x - x_low
    wy_l = 1 - wy_h
    wx_l = 1 - wx_h

    val = 0
    for wx, xp in zip((wx_l, wx_h), (x_low, x_high)):
        for wy, yp in zip((wy_l, wy_h), (y_low, y_high)):
            if 0 <= yp < height and 0 <= xp < width:
                val += wx * wy * data[yp, xp]
    return val

def fn(x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, aligned=False):
    return ipex.nn.modules._roi_align.RoIAlign((pool_h, pool_w), spatial_scale=spatial_scale,
                        sampling_ratio=sampling_ratio, aligned=aligned)(x, rois)

def torchvision_fn(x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, aligned=False):
    return torchvision.ops.RoIAlign((pool_h, pool_w), spatial_scale=spatial_scale,
                        sampling_ratio=sampling_ratio, aligned=aligned)(x, rois)

def expected_fn(in_data, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, aligned=False,
                dtype=torch.float64):

    n_channels = in_data.size(1)
    out_data = torch.zeros(rois.size(0), n_channels, pool_h, pool_w, dtype=dtype)

    offset = 0.5 if aligned else 0.

    for r, roi in enumerate(rois):
        batch_idx = int(roi[0])
        j_begin, i_begin, j_end, i_end = (x.item() * spatial_scale - offset for x in roi[1:])

        roi_h = i_end - i_begin
        roi_w = j_end - j_begin
        bin_h = roi_h / pool_h
        bin_w = roi_w / pool_w

        for i in range(0, pool_h):
            start_h = i_begin + i * bin_h
            grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_h))
            for j in range(0, pool_w):
                start_w = j_begin + j * bin_w
                grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_w))

                for channel in range(0, n_channels):

                    val = 0
                    for iy in range(0, grid_h):
                        y = start_h + (iy + 0.5) * bin_h / grid_h
                        for ix in range(0, grid_w):
                            x = start_w + (ix + 0.5) * bin_w / grid_w
                            val += bilinear_interpolate(in_data[batch_idx, channel, :, :], y, x, snap_border=True)
                    val /= grid_h * grid_w

                    out_data[r, channel, i, j] = val
    return out_data

class RoIAlignTester(TestCase):

    def test_roialign(self):
        pool_size = 5
        # n_channels % (pool_size ** 2) == 0 required for PS opeartions.
        n_channels = 2 * (pool_size ** 2)
        for datatype in [torch.double, torch.float32, torch.float16]:
            x = torch.rand(2, n_channels, 10, 10, dtype=datatype)
            gt_x = x.float().clone().detach().requires_grad_()
            rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                                 [0, 0, 5, 4, 9],
                                 [0, 5, 5, 9, 9],
                                 [1, 0, 0, 9, 9]],
                                dtype=datatype)

            pool_h, pool_w = pool_size, pool_size
            gt_y = expected_fn(gt_x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1)
            gt_y.mean().backward()

            # forward
            with torch.no_grad():
                x0 = x.clone().detach()
                y0 = fn(x0, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1)
            self.assertTrue(y0.dtype == datatype)
            self.assertTrue(torch.allclose(gt_y.to(y0.dtype), y0, rtol=1e-2, atol=1e-2))

            x1 = x.clone().detach().requires_grad_()
            y1 = fn(x1, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1)
            self.assertTrue(y1.dtype == datatype)
            self.assertTrue(torch.allclose(gt_y.to(y1.dtype), y1, rtol=1e-2, atol=1e-2))

            # backward
            y1.mean().backward()
            self.assertTrue(x1.grad.dtype == datatype)
            self.assertTrue(torch.allclose(gt_x.grad.to(x1.dtype), x1.grad, rtol=1e-5, atol=1e-5))

            # test channels last
            x2 = x.clone().detach().to(memory_format=torch.channels_last).requires_grad_()
            y2 = fn(x2, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1)
            self.assertTrue(y2.dtype == datatype)
            self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(torch.allclose(gt_y.to(y2.dtype), y2, rtol=1e-2, atol=1e-2))

            y2.mean().backward()
            self.assertTrue(x2.grad.dtype == datatype)
            self.assertTrue(x2.grad.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(torch.allclose(gt_x.grad.to(x2.dtype), x2.grad, rtol=1e-5, atol=1e-5))

        #test autocast
        with torch.cpu.amp.autocast():
            x3 = x.clone().bfloat16().to(memory_format=torch.channels_last).requires_grad_()
            y3 = fn(x3, rois.bfloat16(), pool_h, pool_w, spatial_scale=1, sampling_ratio=-1)
            y3.mean().backward()
            self.assertTrue(y3.dtype == torch.bfloat16)
            self.assertTrue(torch.allclose(gt_y.to(y3.dtype), y3, rtol=1e-2, atol=1e-2))
            self.assertTrue(x3.grad.dtype == torch.bfloat16)
            self.assertTrue(torch.allclose(gt_x.grad.to(x3.dtype), x3.grad, rtol=1e-5, atol=1e-5))

    @skipIfNoTorchVision
    def test_torchvision_roialign(self):
        pool_size = 5
        # n_channels % (pool_size ** 2) == 0 required for PS opeartions.
        n_channels = 2 * (pool_size ** 2)
        for datatype in [torch.double, torch.float32, torch.float16]:
            x = torch.rand(2, n_channels, 10, 10, dtype=datatype)
            gt_x = x.clone().detach().requires_grad_()
            rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                                 [0, 0, 5, 4, 9],
                                 [0, 5, 5, 9, 9],
                                 [1, 0, 0, 9, 9]],
                                dtype=datatype)

            pool_h, pool_w = pool_size, pool_size
            gt_y = expected_fn(gt_x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1)
            gt_y.mean().backward()

            # forward
            with torch.no_grad():
                x0 = x.clone().detach()
                y0 = torchvision_fn(x0, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1)
            self.assertTrue(y0.dtype == datatype)
            self.assertTrue(torch.allclose(gt_y.to(y0.dtype), y0, rtol=1e-2, atol=1e-2))

            x1 = x.clone().detach().requires_grad_()
            y1 = torchvision_fn(x1, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1)
            self.assertTrue(y1.dtype == datatype)
            self.assertTrue(torch.allclose(gt_y.to(y1.dtype), y1, rtol=1e-2, atol=1e-2))

            y1.mean().backward()
            self.assertTrue(x1.grad.dtype == datatype)
            self.assertTrue(torch.allclose(gt_x.grad.to(x1.dtype), x1.grad, rtol=1e-5, atol=1e-5))

            # test channels last
            x2 = x.clone().detach().to(memory_format=torch.channels_last).requires_grad_()
            y2 = torchvision_fn(x2, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1)
            self.assertTrue(y2.dtype == datatype)
            self.assertTrue(y2.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(torch.allclose(gt_y.to(y2.dtype), y2, rtol=1e-2, atol=1e-2))

            y2.mean().backward()
            self.assertTrue(x2.grad.dtype == datatype)
            self.assertTrue(x2.grad.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(torch.allclose(gt_x.grad.to(x2.dtype), x2.grad, rtol=1e-5, atol=1e-5))

        #test autocast
        with torch.cpu.amp.autocast():
            x3 = x.clone().bfloat16().to(memory_format=torch.channels_last).requires_grad_()
            y3 = torchvision_fn(x3, rois.bfloat16(), pool_h, pool_w, spatial_scale=1, sampling_ratio=-1)
            y3.mean().backward()
            self.assertTrue(y3.dtype == torch.bfloat16)
            self.assertTrue(torch.allclose(gt_y.to(y3.dtype), y3, rtol=1e-2, atol=1e-2))
            self.assertTrue(x3.grad.dtype == torch.bfloat16)
            self.assertTrue(torch.allclose(gt_x.grad.to(x3.dtype), x3.grad, rtol=1e-5, atol=1e-5))


if __name__ == '__main__':
    test = unittest.main()
