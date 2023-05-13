import torch


class autocast(torch.amp.autocast_mode.autocast):
    r"""
    See :class:`torch.autocast`.
    ``torch.xpu.amp.autocast(args...)`` is equivalent to ``torch.autocast("xpu", args...)``
    """

    def __init__(self, enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        super().__init__(
            "xpu", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )
