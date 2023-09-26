import torch
# Note: import order is significant here due to the defect of triton.compile
# in XPU backend. Here codecache is a temp WA.
from torch._inductor import codecache  # noqa
from torch._dynamo.device_interface import register_interface_for_device

from .device_interface import XPUInterface

# Register XPU device interface in PyTorch _dynamo.
if torch.xpu.is_available():
    register_interface_for_device("xpu", XPUInterface)
