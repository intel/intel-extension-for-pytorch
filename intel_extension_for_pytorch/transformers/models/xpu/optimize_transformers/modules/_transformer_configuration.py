import torch
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Union, List


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
    gegelu: str = "gegelu"


@dataclass
class IPEXTransformerConfig:
    num_experts_per_tok: int = 2
    num_local_experts: int = 8
    output_router_logits: bool = False
    embedding_dim: int = 4096
    intermediate_dim: int = None
    num_attention_head: int = 16
    num_key_value_head: int = None
    max_positions: int = 4096
    original_max_position_embeddings: int = 4096
    max_out_positions: int = 256
    kv_channels: int = 128
    rotary_embedding_class: str = "GPTJRotaryEmbedding"
    rope_scaling: Optional[Dict[str, Union[float, List[float], int]]] = None
    rotary_pct: float = 1.0
    rotary_dim: int = 64
    partial_rotary_factor: Optional[float] = 1.0
    rotary_half: bool = False
    rotate_every_two: bool = True
    use_causal_mask: bool = False
    activation_function: str = None
    ipex_act: SupportedActivation = SupportedActivation.gelu
    norm_eps: float = 0.001
    residual_dropout: bool = False
    attn_dropout: bool = False
    enable_bias: bool = False
    residual_pdrop: bool = False
    resid_pdrop: float = 0.0
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
    sliding_window: int = None


@dataclass
class IPEXTransformerConfigChatGLM(IPEXTransformerConfig):
    apply_residual_connection_post_layernorm: bool = False
    multi_query_attention: bool = True
    rmsnorm: bool = True


@dataclass
class IPEXTransformerConfigPhi3Small(IPEXTransformerConfig):
    rope_position_scale: float = 1.0
    # Block Sparse Attention Pattern
    blocksparse_homo_head_pattern: bool = False
    blocksparse_block_size: int = 64
    blocksparse_num_local_blocks: int = 16
    blocksparse_vert_stride: int = 8
    blocksparse_triton_kernel_block_size: int = 64
    # Frequency of block sparsity
    dense_attention_every_n_layers: Optional[int] = 2
    # for gegelu activation
    gegelu_limit: float = 20.0
    # MuP parameters
    mup_use_scaling: bool = True
    mup_attn_multiplier: bool = 1.0
    ffn_dropout_prob: float = 0.1


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
