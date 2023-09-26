import torch 
from dataclasses import dataclass
from enum import Enum


class ImplementMode(Enum):
    naive: int = 0
    optimized: int = 1

class SupportedActivation(Enum):
    gelu: str       = "gelu", "gelu_new", "bloom_gelu"
    relu: str       = "relu"
    sigmoid: str    = "sigmoid"
    silu: str       = "silu"
    tanh: str       = "tanh"
    bloom_gelu: str = "bloom_gelu"

@dataclass
class IPEXTransformerConfig:
    embedding_dim: int = 4096
    intermediate_dim: int = None
    num_attention_head: int = 16
    num_key_value_head: int = None
    max_positions: int = 4096
    max_out_positions: int = 256
    rotary_embedding_class: str = "GPTJRotaryEmbedding"
    rotary_dim: int = 64
    rotary_half: bool = False
    rotate_every_two: bool = True
    rope_scaling = None
    use_casual_mask: bool = False
    activation_function: str = None
    ipex_act: SupportedActivation = SupportedActivation.gelu
    norm_eps: float = 0.001
    residual_dropout: bool = False
    attn_dropout: bool = False
    enable_bias: bool = False
    residual_pdrop: bool = False
    scale_attention: bool = False
    is_decoder: bool = False
    do_norm_before: bool = False
    ln_elementwise_affine: bool = False
    positional_embedding_base: int = 10000
    device : torch.device = "xpu"
    dtype: str = "fp16"
    impl: ImplementMode = ImplementMode.optimized
    beam_num: int = 1
    tp_size: int = 1
    tp_group: object = None
    transpose: bool = True

