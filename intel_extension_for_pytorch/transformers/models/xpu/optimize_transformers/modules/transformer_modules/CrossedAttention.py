import torch
from typing import Optional, Tuple
from .Attention import IPEXTransformerAttnOptimizedFp16
from .._transformer_configuration import IPEXTransformerConfig


class IPEXTransformerAttnOptimizedFp16Crossed(IPEXTransformerAttnOptimizedFp16):
    def __init__(self, config: IPEXTransformerConfig) -> None:
        super().__init__(config)

    def qkv_gemm(
        self,
        hidden_states: torch.Tensor,
        key_value_state: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        **kwargs
    ):
        is_cross_attention = key_value_state is not None
        if is_cross_attention and layer_past is not None:
            query = torch.ops.torch_ipex.matmul_bias_out(
                hidden_states, self.q_proj.weight, self.q_proj.bias
            )
            key = layer_past[0]
            value = layer_past[1]
        else:
            query, key, value = super().qkv_gemm(
                hidden_states, key_value_state, layer_past
            )
        return query, key, value
