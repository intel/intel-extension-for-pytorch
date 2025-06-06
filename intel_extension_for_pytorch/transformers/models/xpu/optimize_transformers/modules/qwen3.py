from .transformer_modules.Norm import QWenRMSNorm


from .transformer_modules.XPUAttentionInt4 import (  # noqa F401
    IPEXAttentionInt4,
    IPEXAttentionInt4OneDNN,
)
from .transformer_modules.Mlp import (  # noqa F401
    IPEXTransformerBaseMLP,
    IPEXTransformerMLPOptimizedFp16,
)

from .transformer_modules.QuantizedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16,
    IPEXTransformerAttnOptimizedInt4,
)  # noqa
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .transformer_modules.GroupedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16Grouped,
)

from .transformer_modules.Mlp import *  # noqa
from .transformer_modules.QuantizedMlp import *  # noqa
from .qwen2 import NewIPEXQWEN2DecoderLayer


class NewIPEXQWEN3DecoderLayer(NewIPEXQWEN2DecoderLayer):
    def __init__(
        self,
        module,
        config,
        dtype="fp16",
        device="xpu",
        module_name="",
        impl_mode=None,
        tp_size=1,
        tp_group=None,
        **kwargs,
    ):
        super().__init__(
            module,
            config,
            dtype,
            device,
            module_name,
            impl_mode,
            tp_size,
            tp_group,
            **kwargs,
        )

        self.q_layernorm = QWenRMSNorm(
            self.ipex_config.head_dim, self.ipex_config.norm_eps
        )
        self.k_layernorm = QWenRMSNorm(
            self.ipex_config.head_dim, self.ipex_config.norm_eps
        )
        self.q_layernorm.weight = self.module.self_attn.q_norm.weight
        self.k_layernorm.weight = self.module.self_attn.k_norm.weight

        self.attn.q_norm = self.q_layernorm
        self.attn.k_norm = self.k_layernorm
