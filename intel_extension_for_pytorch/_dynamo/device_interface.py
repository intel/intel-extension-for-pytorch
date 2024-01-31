from typing import Dict, Union

import torch
from torch._dynamo.device_interface import (
    DeviceInterface,
    caching_worker_current_devices,
    caching_worker_device_properties,
)
from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream

_device_t = Union[torch.device, str, int, None]


class XPUInterface(DeviceInterface):
    device = torch.xpu.device
    Event = torch.xpu.Event
    Stream = torch.xpu.Stream

    class Worker:
        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices["xpu"] = device

        @staticmethod
        def current_device() -> int:
            if "xpu" in caching_worker_current_devices:
                return caching_worker_current_devices["xpu"]
            return torch.xpu.current_device()

        @staticmethod
        def get_device_properties(device: _device_t = None):
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == "xpu"
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = XPUInterface.Worker.current_device()

            if "xpu" not in caching_worker_device_properties:
                device_prop = [
                    torch.xpu.get_device_properties(i)
                    for i in range(torch.xpu.device_count())
                ]
                caching_worker_device_properties["xpu"] = device_prop

            return caching_worker_device_properties["xpu"][device]

    current_device = staticmethod(torch.xpu.current_device)
    set_device = staticmethod(torch.xpu.set_device)
    device_count = staticmethod(torch.xpu.device_count)
    stream = staticmethod(torch.xpu.stream)
    current_stream = staticmethod(torch.xpu.current_stream)
    set_stream = staticmethod(torch.xpu.set_stream)
    _set_stream_by_id = staticmethod(torch.xpu._set_stream_by_id)
    synchronize = staticmethod(torch.xpu.synchronize)
    get_device_properties = staticmethod(torch.xpu.get_device_properties)
    get_raw_stream = staticmethod(get_xpu_stream)

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return torch.xpu.is_available()

    @staticmethod
    def get_compute_capability(device: _device_t = None) -> int:
        # TODO :Return 0x80860001 for ATSM
        # Currently return 0x80860002 for PVC
        # currently, torch.xpu.get_device_capability returns a dict,
        # but we want int for now
        return 86
        # return torch.xpu.get_device_capability(device)
