import torch
from torch.nn.modules.utils import _pair
from torch import nn, Tensor
from torch.jit.annotations import BroadcastingList2
from typing import List, Union
from .modules import Interaction
import intel_extension_for_pytorch

__all__ = [
    "Interaction",
    "nms",
    "locations_to_boxes",
    "roi_align",
]


def MulAdd(input, other, accumu, alpha=1.0):
    return torch.ops.torch_ipex.mul_add(input, other, accumu, alpha)


def nms(dets, scores, iou_threshold):
    return torch.ops.torch_ipex.nms(dets, scores, iou_threshold)


def locations_to_boxes(locations, priors, center_variance, size_variance):
    return torch.ops.torch_ipex.locations_to_boxes(
        locations, priors, center_variance, size_variance
    )


def check_roi_boxes_shape(boxes: Union[Tensor, List[Tensor]]):
    if isinstance(boxes, (list, tuple)):
        for _tensor in boxes:
            torch._assert(
                _tensor.size(1) == 4,
                "The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]",
            )
    elif isinstance(boxes, torch.Tensor):
        torch._assert(
            boxes.size(1) == 5, "The boxes tensor shape is not correct as Tensor[K, 5]"
        )
    else:
        torch._assert(
            False, "boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]"
        )
    return


def convert_boxes_to_roi_format(boxes: List[Tensor]) -> Tensor:
    concat_boxes = _cat(list(boxes), dim=0)
    temp = []
    for i, b in enumerate(boxes):
        temp.append(torch.full_like(b[:, :1], i))
    ids = _cat(temp, dim=0)
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def roi_align(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    check_roi_boxes_shape(boxes)
    rois = boxes
    output_size = _pair(output_size)
    if not isinstance(rois, torch.Tensor):
        rois = convert_boxes_to_roi_format(rois)
    return torch.ops.torch_ipex.roi_align(
        input,
        rois,
        spatial_scale,
        output_size[0],
        output_size[1],
        sampling_ratio,
        aligned,
    )
