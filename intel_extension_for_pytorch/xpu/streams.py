import ctypes
import intel_extension_for_pytorch


class Stream(intel_extension_for_pytorch._C._XPUStreamBase):
    def __new__(cls, device=None, priority=0, **kwargs):
        with intel_extension_for_pytorch.xpu.device(device):
            return super(Stream, cls).__new__(cls, priority=priority, **kwargs)

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.sycl_queue)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super(Stream, self).__eq__(o)
        return False

    def __hash__(self):
        return hash((self.sycl_queue, self.device))

    def __repr__(self):
        return (
            "<intel_extension_for_pytorch.Stream device={0} sycl_queue={1:#x}>".format(
                self.device, self.sycl_queue
            )
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
            event = Event()
        event.record(self)
        return event

    def synchronize(self):
        r"""Wait for all the kernels in this stream to complete."""
        super(Stream, self).synchronize()


class Event(intel_extension_for_pytorch._C._XPUEventBase):
    def __new__(cls, **kwargs):
        return super(Event, cls).__new__(cls, **kwargs)

    def record(self, stream=None):
        r"""Records the event in a given stream.

        Uses ``intel_extension_for_pytorch.xpu.current_stream()`` if no stream is specified.
        """
        if stream is None:
            stream = intel_extension_for_pytorch.xpu.current_stream()
        super(Event, self).record(stream)

    def wait(self, stream=None):
        r"""Makes all future work submitted to the given stream wait for this
        event.

        Use ``intel_extension_for_pytorch.xpu.current_stream()`` if no stream is specified.
        """
        if stream is None:
            stream = intel_extension_for_pytorch.xpu.current_stream()
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
            return "<torch.xpu.Event {0:#x}>".format(self._as_parameter_.value)
        else:
            return "<torch.xpu.Event uninitialized>"
