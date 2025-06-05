from .optimize import optimize_transformers
from .optimize import _set_optimized_model_for_generation
from .models.cpu.modules.attentions import _IPEXAttentionCPU
from .models.cpu.modules.decoder import _IPEXDecoderLayerCPU

try:
    from .models.reference.models import detect_language
except ImportError:
    pass
