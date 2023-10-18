from torch import nn, Tensor
from torch.jit.annotations import BroadcastingList2

from ...nn import functional as F


class RoIAlign(nn.Module):
    """
    See :func:`roi_align`.
    """

    def __init__(
        self,
        output_size: BroadcastingList2[int],
        spatial_scale: float,
        sampling_ratio: int,
        aligned: bool = False,
    ):
        super(RoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        return F._roi_align_helper.roi_align(
            input,
            rois,
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            self.aligned,
        )

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr
