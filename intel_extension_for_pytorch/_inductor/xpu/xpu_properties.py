import torch


def current_device():
    from torch._inductor.cuda_properties import _compile_worker_current_device
    if _compile_worker_current_device is not None:
        return _compile_worker_current_device
    return torch.xpu.current_device()
