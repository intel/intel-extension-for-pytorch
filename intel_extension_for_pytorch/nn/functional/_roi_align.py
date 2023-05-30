# This Python file uses the following encoding: utf-8

from typing import List, Union

import torch
from torch import Tensor
from torch.nn.modules.utils import _pair
from torch.jit.annotations import BroadcastingList2


def _cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    # TODO add back the assert
    # assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def _convert_boxes_to_roi_format(boxes: List[Tensor]) -> Tensor:
    concat_boxes = _cat(list(b for b in boxes), dim=0)
    temp = []
    for i, b in enumerate(boxes):
        temp.append(torch.full_like(b[:, :1], i))
    ids = _cat(temp, dim=0)
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def _check_roi_boxes_shape(boxes: Union[Tensor, List[Tensor]]):
    if isinstance(boxes, (list, tuple)):
        for _tensor in boxes:
            assert (
                _tensor.size(1) == 4
            ), "The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]"
    elif isinstance(boxes, torch.Tensor):
        assert (
            boxes.size(1) == 5
        ), "The boxes tensor shape is not correct as Tensor[K, 5]"
    else:
        raise ValueError(
            "boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]"
        )
    return


def roi_align(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    """
    Performs Region of Interest (RoI) Align operator with average pooling, as described in Mask R-CNN.
    It is optimized with parallelization and channels last support on the basis of the torchvision's roi_align.

    The semantics of Intel® Extension for PyTorch* roi_align is exactly the same as that of torchvision.
    We override roi_align in the torchvision with Intel® Extension for PyTorch* roi_align via ATen op registration.
    It is activated when Intel® Extension for PyTorch* is imported from the Python frontend or when it is linked
    by a C++ program. It is fully transparent to users.

    In certain cases, if you are using a self-implemented roi_align class or function that behave exactly the same as
    the ones in torchvision, please import the optimized one in Intel® Extension for PyTorch* as the following
    examples to get performance boost on Intel platforms.

    .. highlight:: python
    .. code-block:: python

        from intel_extension_for_pytorch import roi_align

    or

    .. highlight:: python
    .. code-block:: python

        from intel_extension_for_pytorch import RoIAlign

    Args:
        input (Tensor[N, C, H, W]): The input tensor, i.e. a batch with ``N`` elements. Each element
            contains ``C`` feature maps of dimensions ``H x W``.
            If the tensor is quantized, we expect a batch size of ``N == 1``.
        boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
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
    _check_roi_boxes_shape(boxes)
    rois = boxes
    output_size = _pair(output_size)
    if not isinstance(rois, torch.Tensor):
        rois = _convert_boxes_to_roi_format(rois)
    return torch.ops.torch_ipex.ROIAlign_forward(
        input,
        rois,
        spatial_scale,
        output_size[0],
        output_size[1],
        sampling_ratio,
        aligned,
    )
