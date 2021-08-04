import ctypes
import torch_ipex


class Stream(torch_ipex._C._XPUStreamBase):
    def __new__(cls, device=None, priority=0, **kwargs):
        with torch_ipex.device(device):
            return super(Stream, cls).__new__(cls, priority=priority, **kwargs)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super(Stream, self).__eq__(o)
        return False

    def __hash__(self):
        return hash((self._cdata, self.device))

    def __repr__(self):
        return ('<torch_ipex.Stream device={0} xpu_stream={1}>'
                .format(self.device, self.xpu_stream))


class Event(torch_ipex._C._XPUEventBase):
    def __new__(cls, **kwargs):
        return super(Event, cls).__new__(cls, **kwargs)

    def record(self, stream=None):
        r"""Records the event in a given stream.

        Uses ``torch_ipex.current_stream()`` if no stream is specified."""
        if stream is None:
            stream = torch_ipex.current_stream()
        super(Event, self).record(stream)

    def wait(self, stream=None):
        r"""Makes all future work submitted to the given stream wait for this
        event.

        Use ``torch_ipex.current_stream()`` if no stream is specified."""
        if stream is None:
            stream = torch_ipex.current_stream()
        super(Event, self).wait(stream)

    def query(self):
        r"""Checks if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super(Event, self).query()

    def elapsed_time(self, end_event):
        r"""Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
        return super(Event, self).elapsed_time(end_event)

    def synchronize(self):
        r"""Waits for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        super(Event, self).synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.dpcpp_event)

    def __repr__(self):
        if self.dpcpp_event:
            return '<torch.xpu.Event {0:#x}>'.format(self._as_parameter_.value)
        else:
            return '<torch.xpu.Event uninitialized>'
