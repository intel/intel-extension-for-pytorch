import torch
import intel_extension_for_pytorch._C as core

VERBOSE_OFF = 0
VERBOSE_ON = 1
VERBOSE_ON_CREATION = 2


class verbose(object):
    """
    On-demand oneDNN verbosing functionality

    To make it easier to debug performance issues, oneDNN can dump verbose
    messages containing information like kernel size, input data size and
    execution duration while executing the kernel. The verbosing functionality
    can be invoked via an environment variable named `DNNL_VERBOSE`. However,
    this methodology dumps messages in all steps. Those are a large amount of
    verbose messages. Moreover, for investigating the performance issues,
    generally taking verbose messages for one single iteration is enough.

    This on-demand verbosing functionality makes it possible to control scope
    for verbose message dumping. In the following example, verbose messages
    will be dumped out for the second inference only.

    .. highlight:: python
    .. code-block:: python

        import intel_extension_for_pytorch as ipex
        model(data)
        with ipex.verbose(ipex.VERBOSE_ON):
            model(data)

    Args:
        level: Verbose level

            - ``VERBOSE_OFF``: Disable verbosing
            - ``VERBOSE_ON``:  Enable verbosing
            - ``VERBOSE_ON_CREATION``: Enable verbosing, including oneDNN kernel creation

    :meta public:
    """

    def __init__(self, level):
        self.level = level

    def __enter__(self):
        if self.level == VERBOSE_OFF:
            return
        try:
            st = torch._C._verbose.mkldnn_set_verbose(self.level)
            assert bool(
                st
            ), "Failed to set Verbose mode of MKLDNN in PyTorch. Please consider to disable this verbose scope."
        except BaseException:
            pass
        st = core.mkldnn_set_verbose(self.level)
        assert bool(
            st
        ), "Failed to set Verbose mode of MKLDNN in IPEX. Please consider to disable this verbose scope."
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        core.mkldnn_set_verbose(VERBOSE_OFF)
        try:
            torch._C._verbose.mkldnn_set_verbose(VERBOSE_OFF)
        except BaseException:
            pass
        return False


try:
    verbose_torch = torch.backends.mkldnn.verbose
    torch.backends.mkldnn.verbose = verbose
except BaseException:
    pass
