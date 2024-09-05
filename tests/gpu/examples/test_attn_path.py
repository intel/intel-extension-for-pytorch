import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

from intel_extension_for_pytorch.transformers.models.xpu.optimize_transformers.modules.transformer_modules import (
    Attention,
    CrossedAttention,
    GroupedAttention,
    QuantizedAttention,
)

from intel_extension_for_pytorch.transformers.models.xpu.optimize_transformers.modules._transformer_configuration import (
    ImplementMode,
    IPEXTransformerConfig,
)
from intel_extension_for_pytorch.transformers.models.xpu.optimize_transformers.modules.transformer_modules.DecoderBlock import (
    IPEXTransformerBlock,
)

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


ATTN_CLASS = {
    Attention.IPEXTransformerAttnOptimizedFp16: (
        "fp16",
        ImplementMode.optimized,
        None,
        False,
    ),
    Attention.IPEXTransformerAttnOptimizedFp16Baichuan: (
        "fp16",
        ImplementMode.optimized,
        "Baichuan",
        False,
    ),
    CrossedAttention.IPEXTransformerAttnOptimizedFp16Crossed: (
        "fp16",
        ImplementMode.optimized,
        "Crossed",
        False,
    ),
    GroupedAttention.IPEXTransformerAttnOptimizedFp16Grouped: (
        "fp16",
        ImplementMode.optimized,
        None,
        True,
    ),
    GroupedAttention.IPEXTransformerAttnOptimizedInt4Grouped: (
        "int4",
        ImplementMode.optimized,
        None,
        True,
    ),
    QuantizedAttention.IPEXTransformerAttnOptimizedInt4: (
        "int4",
        ImplementMode.optimized,
        None,
        False,
    ),
}


class TestTorchMethod(TestCase):
    def test_Attn_path(self, dtype=torch.float16, device=dpcpp_device):
        IPEXBlock = IPEXTransformerBlock(None, None, None, None, None)
        for attn in ATTN_CLASS.keys():
            attn_config = ATTN_CLASS[attn]
            IPEXBlock.ipex_config = IPEXTransformerConfig(
                dtype=attn_config[0], impl=attn_config[1], num_key_value_head=128
            )
            attn_type = IPEXBlock.build_attention_from_config(
                attn_config[2], attn_config[3]
            )
            self.assertIsInstance(attn_type, attn)
