import ctypes
import torch_ipex


class Stream(torch_ipex._C._DPCPPStreamBase):
    def __new__(cls, device=None, priority=0, **kwargs):
        with torch_ipex.device(device):
            return super(Stream, cls).__new__(cls, priority=priority, **kwargs)

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.xpu_stream)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super(Stream, self).__eq__(o)
        return False

    def __hash__(self):
        return hash((self.dpcpp_stream, self.device))

    def __repr__(self):
        return ('<torch_ipex.Stream device={0} dpcpp_stream={1:#x}>'
                .format(self.device, self.dpcpp_stream))


class Event:
    pass
