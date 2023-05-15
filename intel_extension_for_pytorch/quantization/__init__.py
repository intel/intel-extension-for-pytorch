from ._quantize import prepare, convert
from ._qconfig import (
    default_static_qconfig,
    default_dynamic_qconfig,
    default_static_qconfig_mapping,
    default_dynamic_qconfig_mapping,
    get_smooth_quant_qconfig_mapping,
    weight_only_quant_qconfig_mapping,
)
from ._autotune import autotune
