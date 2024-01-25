from .optimize import optimize_transformers
from .optimize import _set_optimized_model_for_generation
from ..utils.utils import has_cpu

if has_cpu():
    from .models.cpu.modules.attentions import _IPEXAttentionCPU
    from .models.cpu.modules.decoder import _IPEXDecoderLayerCPU
