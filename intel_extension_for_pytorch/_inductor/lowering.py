import torch
from .xpu_properties import current_device


def decode_device(device):
    if device is None:
        return torch.tensor(0.0).device  # default device
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "xpu" and device.index is None:
        return torch.device("xpu", index=current_device())
    return device
