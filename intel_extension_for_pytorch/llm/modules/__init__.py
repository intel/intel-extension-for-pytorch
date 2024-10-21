from .linear_fusion import (
    LinearSilu,
    LinearSiluMul,
    Linear2SiluMul,
    LinearRelu,
    LinearNewGelu,
    LinearGelu,
    LinearMul,
    LinearAdd,
    LinearAddAdd,
    GatedMLPMOE,
    LinearMOE,
    LinearMOETP
)
from .mha_fusion import (
    RotaryEmbedding,
    RMSNorm,
    FastLayerNorm,
    IndirectAccessKVCacheAttention,
    PagedAttention,
    VarlenAttention,
)
