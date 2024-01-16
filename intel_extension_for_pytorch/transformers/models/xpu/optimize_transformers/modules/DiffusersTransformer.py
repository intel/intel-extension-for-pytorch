from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from ._transformer_configuration import IPEXDiffusersTransformerConfig
from .transformer_modules.Decoderblock import IPEXTransformerBlock
from .transformer_modules.DiffusersAttention import DiffusersAttention
from .transformer_modules.lora import LoRACompatibleLinear


class GatedSelfAttentionDense(nn.Module):
    r"""
    A gated self-attention dense layer that combines visual features and object features.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        context_dim (`int`): The number of channels in the context.
        n_heads (`int`): The number of heads to use for attention.
        d_head (`int`): The number of channels in each head.
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        self.enabled = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        n_visual = x.shape[1]
        objs = self.linear(objs)

        x = (
            x
            + self.alpha_attn.tanh()
            * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :]
        )
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        return x


class NewIPEXBasicTransformerBlock(IPEXTransformerBlock):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

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
    ):
        #     self,
        #     dim: int,
        #     num_attention_heads: int,
        #     attention_head_dim: int,
        #     dropout=0.0,
        #     cross_attention_dim: Optional[int] = None,
        #     activation_fn: str = "geglu",
        #     num_embeds_ada_norm: Optional[int] = None,
        #     attention_bias: bool = False,
        #     only_cross_attention: bool = False,
        #     double_self_attention: bool = False,
        #     upcast_attention: bool = False,
        #     norm_elementwise_affine: bool = True,
        #     norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
        #     norm_eps: float = 1e-5,
        #     final_dropout: bool = False,
        #     attention_type: str = "default",
        #     positional_embeddings: Optional[str] = None,
        #     num_positional_embeddings: Optional[int] = None,
        # ):
        #     super().__init__()
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(
            config, device, dtype, impl_mode, tp_size, tp_group
        )

        self.dim = (
            self.ipex_config.num_attention_heads * self.ipex_config.attention_head_dim
        )

        self.only_cross_attention = self.ipex_config.only_cross_attention

        self.use_ada_layer_norm_zero = (
            self.ipex_config.num_embeds_ada_norm is not None
        ) and self.ipex_config.norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (
            self.ipex_config.num_embeds_ada_norm is not None
        ) and self.ipex_config.norm_type == "ada_norm"
        self.use_ada_layer_norm_single = self.ipex_config.norm_type == "ada_norm_single"
        self.use_layer_norm = self.ipex_config.norm_type == "layer_norm"

        if (
            self.ipex_config.norm_type in ("ada_norm", "ada_norm_zero")
            and self.ipex_config.num_embeds_ada_norm is None
        ):
            raise ValueError(
                "`norm_type` is set to {self.ipex_config.norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {self.ipex_config.norm_type}."
            )

        if self.ipex_config.positional_embeddings and (
            self.ipex_config.num_positional_embeddings is None
        ):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if self.ipex_config.positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                self.dim,
                max_seq_length=self.ipex_config.num_positional_embeddings,
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(self.dim, self.ipex_config.num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(
                self.dim, self.ipex_config.num_embeds_ada_norm
            )
        else:
            self.norm1 = nn.LayerNorm(
                self.dim,
                elementwise_affine=self.ipex_config.norm_elementwise_affine,
                eps=self.ipex_config.norm_eps,
            )

        self.attn1 = DiffusersAttention(
            query_dim=self.dim,
            heads=self.ipex_config.num_attention_heads,
            dim_head=self.ipex_config.attention_head_dim,
            dropout=self.ipex_config.dropout,
            bias=self.ipex_config.attention_bias,
            cross_attention_dim=self.ipex_config.cross_attention_dim
            if self.ipex_config.only_cross_attention
            else None,
            upcast_attention=self.ipex_config.upcast_attention,
        )

        # 2. Cross-Attn
        if (
            self.ipex_config.cross_attention_dim is not None
            or self.ipex_config.double_self_attention
        ):
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = (
                AdaLayerNorm(self.dim, self.ipex_config.num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(
                    self.dim,
                    elementwise_affine=self.ipex_config.norm_elementwise_affine,
                    eps=self.ipex_config.norm_eps,
                )
            )
            self.attn2 = DiffusersAttention(
                query_dim=self.dim,
                cross_attention_dim=self.ipex_config.cross_attention_dim
                if not self.ipex_config.double_self_attention
                else None,
                heads=self.ipex_config.num_attention_heads,
                dim_head=self.ipex_config.attention_head_dim,
                dropout=self.ipex_config.dropout,
                bias=self.ipex_config.attention_bias,
                upcast_attention=self.ipex_config.upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if not self.use_ada_layer_norm_single:
            self.norm3 = nn.LayerNorm(
                self.dim,
                elementwise_affine=self.ipex_config.norm_elementwise_affine,
                eps=self.ipex_config.norm_eps,
            )

        self.ff = FeedForward(
            self.dim,
            dropout=self.ipex_config.dropout,
            activation_fn=self.ipex_config.activation_fn,
            final_dropout=self.ipex_config.final_dropout,
        )

        # 4. Fuser
        if (
            self.ipex_config.attention_type == "gated"
            or self.ipex_config.attention_type == "gated-text-image"
        ):
            self.fuser = GatedSelfAttentionDense(
                self.dim,
                self.ipex_config.cross_attention_dim,
                self.ipex_config.num_attention_heads,
                self.ipex_config.attention_head_dim,
            )

        # 5. Scale-shift for PixArt-Alpha.
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = nn.Parameter(
                torch.randn(6, self.dim) / self.dim**0.5
            )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        self.port_all_parameters_to_new_module()

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def port_attn_parameter(self):
        self.attn1.load_parameter(
            self.module.attn1.to_q,
            self.module.attn1.to_k,
            self.module.attn1.to_v,
            self.module.attn1.to_out[0],
        )
        self.attn2.load_parameter(
            self.module.attn2.to_q,
            self.module.attn2.to_k,
            self.module.attn2.to_v,
            self.module.attn2.to_out[0],
        )
        self.ff.load_parameter(
            self.module.ff.net,
        )

    def port_mlp_parameter(self):
        pass

    def port_norm_parameter(self):
        self.norm1.weight = self.module.norm1.weight
        self.norm1.bias = self.module.norm1.bias
        self.norm2.weight = self.module.norm2.weight
        self.norm2.bias = self.module.norm2.bias
        self.norm3.weight = self.module.norm3.weight
        self.norm3.bias = self.module.norm3.bias

    def port_all_parameters_to_new_module(self):
        super().port_all_parameters_to_new_module()
        self.attn1.cat_qkv()

    def build_ipex_transformer_config(
        self, config, device, dtype, impl_mode, tp_size, tp_group
    ) -> IPEXDiffusersTransformerConfig:
        assert dtype in [
            "fp16",
            "int4",
        ], "dtype tag {} passed to optimized_transformers is not supported!".format(
            dtype
        )

        return IPEXDiffusersTransformerConfig(
            num_attention_heads=self.config.num_attention_heads,
            attention_head_dim=self.config.attention_head_dim,
            dropout=self.config.dropout,
            cross_attention_dim=self.config.cross_attention_dim,
            activation_fn=self.config.activation_fn,
            num_embeds_ada_norm=self.config.num_embeds_ada_norm,
            attention_bias=self.config.attention_bias,
            only_cross_attention=self.config.only_cross_attention,
            double_self_attention=self.config.double_self_attention,
            upcast_attention=self.config.upcast_attention,
            norm_elementwise_affine=self.config.norm_elementwise_affine,
            norm_type=self.config.norm_type,
            norm_eps=self.config.norm_eps,
            final_dropout=False,
            attention_type=self.config.attention_type,
            positional_embeddings=None,
            num_positional_embeddings=None,
            dtype=dtype,
            tp_size=tp_size,
            tp_group=tp_group,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            # norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = torch.ops.torch_ipex.fast_layer_norm(
                hidden_states,
                self.norm1.normalized_shape,
                self.norm1.weight,
                self.norm1.bias,
                self.norm1.eps,
            )
        elif self.use_ada_layer_norm_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention
            else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.use_ada_layer_norm_single:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.use_layer_norm:
                norm_hidden_states = torch.ops.torch_ipex.fast_layer_norm(
                    hidden_states,
                    self.norm2.normalized_shape,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.norm2.eps,
                )
            elif self.use_ada_layer_norm_single:
                norm_hidden_states = hidden_states
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if not self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        if self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    "`hidden_states` dimension to be chunked: "
                    f"{norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}."
                    f"Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [
                    self.ff(hid_slice, scale=lora_scale)
                    for hid_slice in norm_hidden_states.chunk(
                        num_chunks, dim=self._chunk_dim
                    )
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear_cls = LoRACompatibleLinear

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(linear_cls(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def load_parameter(self, net):
        self.net[2].weight = net[2].weight
        self.net[2].bias = net[2].bias

        self.net[0].load_parameter(
            self.net[0],
        )

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        compatible_cls = (GEGLU, LoRACompatibleLinear)
        for module in self.net:
            if isinstance(module, compatible_cls):
                hidden_states = module(hidden_states, scale)
            else:
                hidden_states = module(hidden_states)
        return hidden_states


class SinusoidalPositionalEmbedding(nn.Module):
    """Apply positional information to a sequence of embeddings.

    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them

    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings

    """

    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length]
        return x


class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()

        self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.LongTensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(
            self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype))
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
            6, dim=1
        )
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def load_parameter(self, net):
        self.proj.weight = net.proj.weight
        self.proj.bias = net.proj.bias
        print("GELU net.proj.weight", net.proj.weight)
        print("GELU self.proj.weight", self.proj.weight)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(
            dtype=gate.dtype
        )

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        linear_cls = LoRACompatibleLinear

        self.proj = linear_cls(dim_in, dim_out * 2)

    def load_parameter(self, net):
        self.proj.weight.data = net.proj.weight.to("xpu").to(torch.float16)
        self.proj.bias.data = net.proj.bias.to("xpu").to(torch.float16)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states, scale: float = 1.0):
        args = (scale,)
        hidden_states, gate = self.proj(hidden_states, *args).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    r"""
    The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://arxiv.org/abs/1606.08415).

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def load_parameter(self, net):
        self.proj.weight = net.proj.weight
        self.proj.bias = net.proj.bias
        print("ApproximateGELU net.proj.weight", net.proj.weight)
        print("ApproximateGELU self.proj.weight", self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)
