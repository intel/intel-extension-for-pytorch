import torch
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ImplementMode(Enum):
    naive: int = 0
    optimized: int = 1


class SupportedActivation(Enum):
    gelu: str = "gelu", "gelu_new", "bloom_gelu"
    relu: str = "relu"
    sigmoid: str = "sigmoid"
    silu: str = "silu"
    tanh: str = "tanh"
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
    device: torch.device = "xpu"
    dtype: str = "fp16"
    impl: ImplementMode = ImplementMode.optimized
    beam_num: int = 1
    tp_size: int = 1
    tp_group: object = None
    transpose: bool = True
    dynamic_cache_stride: int = 16


@dataclass
class IPEXDiffusersTransformerConfig:
    num_attention_heads: int
    attention_head_dim: int
    dropout: float = 0.0
    cross_attention_dim: Optional[int] = None
    activation_fn: str = "geglu"
    num_embeds_ada_norm: Optional[int] = None
    attention_bias: bool = False
    only_cross_attention: bool = False
    double_self_attention: bool = False
    upcast_attention: bool = False
    norm_elementwise_affine: bool = True
    norm_type: str = (
        "layer_norm"  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
    )
    norm_eps: float = 1e-5
    final_dropout: bool = False
    attention_type: str = "default"
    positional_embeddings: Optional[str] = None
    num_positional_embeddings: Optional[int] = None
    device: torch.device = "xpu"
    dtype: str = "fp16"
    tp_size: int = 1
    tp_group: object = None
