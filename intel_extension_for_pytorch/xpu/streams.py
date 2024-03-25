import ctypes
from torch._streambase import  _StreamBase
import intel_extension_for_pytorch
from ..utils.capsule import get_pointer_from_capsule


class Stream(intel_extension_for_pytorch._C._XPUStreamBase, _StreamBase):
    def __new__(cls, device=None, priority=0, **kwargs):
        # setting device manager is expensive, so we avoid it unless necessary
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            return super(Stream, cls).__new__(cls, priority=priority, **kwargs)
        else:
            with intel_extension_for_pytorch.xpu.device(device):
                return super(Stream, cls).__new__(cls, priority=priority, **kwargs)

    @property
    def sycl_queue(self):
        r"""sycl_queue(self): -> PyCapsule

        Returns the sycl queue of the corresponding Stream in a ``PyCapsule``, which encapsules
        a void pointer address. Its capsule name is ``torch.xpu.Stream.sycl_queue``.
        """
        return super(Stream, self).sycl_queue

    @property
    def _as_parameter_(self):
        r"""Return the sycl queue void pointer address. Make it be easily used in
        C/C++ code.
        """
        return ctypes.c_void_p(get_pointer_from_capsule(self.sycl_queue))

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super(Stream, self).__eq__(o)
        return False

    def __hash__(self):
        return hash((self.sycl_queue, self.device))

    def __repr__(self):
        return "<torch.xpu.Stream device={0} sycl_queue={1}>".format(
            self.device, self.sycl_queue
        )

    def wait_event(self, event):
        r"""Makes all future work submitted to the stream wait for an event.

        Arguments:
            event (Event): an event to wait for.
        """
        event.wait(self)

    def wait_stream(self, stream):
        r"""Synchronizes with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Arguments:
            stream (Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        r"""Records an event.

        Arguments:
            event (Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            #TODO: [Rebase] Should delete this after rebasing stream.
            pass
            # event = Event()
        event.record(self)
        return event

    def synchronize(self):
        r"""Wait for all the kernels in this stream to complete."""
        super(Stream, self).synchronize()


