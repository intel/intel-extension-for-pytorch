import torch
from torch._inductor.cuda_properties import current_device as cuda_current_device
from .xpu_properties import current_device as xpu_current_device


def decode_device(device):
    if device is None:
        return torch.tensor(0.0).device  # default device
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", index=cuda_current_device())
    if device.type == "xpu" and device.index is None:
        return torch.device("xpu", index=xpu_current_device())
    return device
