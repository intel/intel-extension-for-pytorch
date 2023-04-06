import torch
from torch.ao.quantization import (
    PlaceholderObserver,
    PerChannelMinMaxObserver,
    HistogramObserver,
    QConfig,
    QConfigMapping,
)


_default_weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)

default_static_qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                                 weight=_default_weight_observer)
"""
Default qconfig configuration for static quantization.
"""

default_static_qconfig_mapping = QConfigMapping().set_global(default_static_qconfig)

default_dynamic_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float, is_dynamic=True),
                                  weight=_default_weight_observer)
"""
Default qconfig configuration for dynamic quantization.
"""

default_dynamic_qconfig_mapping = QConfigMapping().set_global(default_dynamic_qconfig)
