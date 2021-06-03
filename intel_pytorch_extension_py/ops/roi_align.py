# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair


class _ROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio, aligned):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.aligned = aligned
        ctx.input_shape = input.size()
        output = torch.ops.torch_ipex.ROIAlign_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        aligned = ctx.aligned
        bs, ch, h, w = ctx.input_shape
        grad_input = torch.ops.torch_ipex.ROIAlign_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
            aligned
        )
        return grad_input, None, None, None, None, None


roi_align = _ROIAlign.apply


class ROIAlign(nn.Module):
    """
    Performs Region of Interest (RoI) Align operator with average pooling, as described in Mask R-CNN.
    Args:
        input (Tensor[N, C, H, W]): The input tensor, i.e. a batch with ``N`` elements. Each element
            contains ``C`` feature maps of dimensions ``H x W``.
            If the tensor is quantized, we expect a batch size of ``N == 1``.
        rois (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
            format where the regions will be taken from.
            The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            If a single Tensor is passed, then the first column should
            contain the index of the corresponding element in the batch, i.e. a number in ``[0, N - 1]``.
            If a list of Tensors is passed, then each Tensor will correspond to the boxes for an element i
            in the batch.
        output_size (int or Tuple[int, int]): the size of the output (in bins or pixels) after the pooling
            is performed, as (height, width).
        spatial_scale (float): a scaling factor that maps the input coordinates to
            the box coordinates. Default: 1.0
        sampling_ratio (int): number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly ``sampling_ratio x sampling_ratio`` sampling points per bin are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ``ceil(roi_width / output_width)``, and likewise for height). Default: -1
        aligned (bool): If False, use the legacy implementation.
            If True, pixel shift the box coordinates it by -0.5 for a better alignment with the two
            neighboring pixel indices. This version is used in Detectron2
    Returns:
        Tensor[K, C, output_size[0], output_size[1]]: The pooled RoIs.
    """
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned = False):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input, rois):
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ', aligned=' + str(self.aligned)
        tmpstr += ")"
        return tmpstr
