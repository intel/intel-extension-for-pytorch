from typing import Tuple, Union
from typing_extensions import deprecated

import torch
from torch.types import Device


@deprecated(
    "intel_extension_for_pytorch.xpu.mem_get_info is deprecated and will be removed in a future release. "
    "Please use torch.xpu.mem_get_info() instead.",
    category=FutureWarning,
)
def mem_get_info(device: Union[Device, int] = None) -> Tuple[int, int]:
    r"""Return the estimated value of global free and total GPU memory for a given device.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default) or if the device index is not specified.

    .. note::
        See :ref:`xpu-memory-management` for more details about GPU memory
        management.
    """
    return torch.xpu.mem_get_info(device)
