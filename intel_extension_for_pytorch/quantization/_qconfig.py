import torch
from torch.ao.quantization import PlaceholderObserver, PerChannelMinMaxObserver, HistogramObserver, QConfig


_default_weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)

default_static_qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                                 weight=_default_weight_observer)
"""
Default qconfig configuration for static quantization.
"""

default_dynamic_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float, compute_dtype=torch.quint8),
                                  weight=_default_weight_observer)
"""
Default qconfig configuration for dynamic quantization.
"""
