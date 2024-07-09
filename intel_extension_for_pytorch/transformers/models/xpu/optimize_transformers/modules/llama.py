import torch
from typing import Optional, Tuple, Dict, Any
from .transformer_modules.RoPE import LlamaRotaryEmbedding
from .transformer_modules.Norm import LlamaRMSNorm


from ._transformers import MAX_SEQ_LEN, MAX_OUT_SEQ_LEN
from .transformer_modules.BaseAttention import IPEXTransformerAttn
from ._transformer_configuration import IPEXTransformerConfig, SupportedActivation
from .transformer_modules.QuantizedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16,
    IPEXTransformerAttnOptimizedInt4,
)  # noqa
from .transformer_modules.NaiveAttention import IPEXTransformerAttnNaive  # noqa
from .transformer_modules.GroupedAttention import (  # noqa F401
    IPEXTransformerAttnOptimizedFp16Grouped,
    IPEXTransformerAttnOptimizedInt4Grouped,
)
from .transformer_modules.DecoderBlock import IPEXTransformerBlock
from .transformer_modules.Mlp import *  # noqa
from .transformer_modules.QuantizedMlp import *  # noqa
from transformers.cache_utils import StaticCache

import os

acc_test = os.environ.get("LLM_ACC_TEST", "OFF").upper() in [
    "1",
    "ON",
    "Y",
    "YES",
    "TRUE",
]


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    #Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    #Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


# `_update_causal_mask` in Transformers 4.38.1 is designed for StaticCache which is not use in IPEX.
# Replace it with `_prepare_decoder_attention_mask` in Transformers 4.31.0 to avoid performance overhead.
def _update_causal_mask(self, attention_mask, input_tensor):
    batch_size, seq_length = input_tensor.shape[:2]
    dtype = input_tensor.dtype
    device = input_tensor.device

    input_shape = (batch_size, seq_length)
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            dtype,
            device,
            0,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, dtype, tgt_len=input_shape[-1]
        ).to(device)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


class LlamaCache(StaticCache):
    """p
    A cache that only keeps the attributes of past_key_value to avoid redundant kernels in the default Cache, `DynamicCache`.

    It stores tensors with the same shape as the Key and Value states, but the content is meaningless.
    The expected shape for each tensor is `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: torch.Tensor = None
        self.value_cache: torch.Tensor = None
        self.seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )
        self.len = 0

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache, self.value_cache)
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        raise KeyError("Cache only save the shape of key and value.")

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return self.len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Updates the attributes that the cache stores.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self.seen_tokens = key_states.shape[-2]

        # Update the cache
        self.len = max(self.len, layer_idx + 1)

        self.key_cache = key_states
        self.value_cache = value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache.shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. LlamaCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        device = self.key_cache.device
        self.key_cache = self.key_cache.index_select(0, beam_idx.to(device))
        device = self.value_cache.device
        self.value_cache = self.value_cache.index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `LlamaCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache, self.value_cache),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "LlamaCache":
        """Converts a cache in the legacy cache format into an equivalent `LlamaCache`."""
        cache = cls()
        if past_key_values is not None:
            layer_idx = len(past_key_values) - 1
            key_states, value_states = past_key_values[0]
            cache.update(key_states, value_states, 0)
            key_states, value_states = past_key_values[layer_idx]
            cache.update(key_states, value_states, layer_idx)
        return cache


class NewIPEXLLAMABlock(IPEXTransformerBlock):
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
        super().__init__(module, config, dtype, device, module_name)
        self.ipex_config = self.build_ipex_transformer_config(
            config, device, dtype, impl_mode, tp_size, tp_group
        )
        grouped = False
        if self.ipex_config.num_attention_head > self.ipex_config.num_key_value_head:
            grouped = True
        self.self_attn = self.build_attention_from_config(grouped=grouped)
        self.mlp = self.build_mlp_from_config("llama")
        self.input_layernorm = LlamaRMSNorm(
            self.ipex_config.embedding_dim, self.ipex_config.norm_eps
        )
        self.post_attn_layernorm = LlamaRMSNorm(
            self.ipex_config.embedding_dim, self.ipex_config.norm_eps
        )
        self.port_all_parameters_to_new_module()
        self.layer_idx = kwargs.get("layer_idx", None)

    def build_ipex_transformer_config(
        self, config, device, dtype, impl_mode, tp_size, tp_group
    ) -> IPEXTransformerConfig:
        activation_function = self.config.hidden_act
        ipex_activation = None
        for act in SupportedActivation:
            if activation_function in act.value:
                ipex_activation = act
                break
        assert ipex_activation is not None, (
            "found unrecognized activation function,"
            "can not build ipex config from {}".format(activation_function)
        )

        assert dtype in [
            "fp16",
            "int4",
        ], "dtype tag {} passed to optimized_transformers is not supported!".format(
            dtype
        )

        return IPEXTransformerConfig(
            embedding_dim=self.config.hidden_size,
            intermediate_dim=self.config.intermediate_size,
            num_attention_head=self.config.num_attention_heads,
            # transformers==4.31.0
            num_key_value_head=self.config.num_key_value_heads,
            max_positions=max(self.config.max_position_embeddings, MAX_SEQ_LEN),
            max_out_positions=MAX_OUT_SEQ_LEN,
            rotary_embedding_class=LlamaRotaryEmbedding,
            rotary_dim=None,
            rotary_half=True,
            rotate_every_two=False,
            use_casual_mask=False,
            activation_function=self.config.hidden_act,
            ipex_act=ipex_activation,
            norm_eps=self.config.rms_norm_eps,
            residual_dropout=None,
            attn_dropout=None,
            enable_bias=False,
            residual_pdrop=None,
            scale_attention=True,
            is_decoder=False,
            do_norm_before=None,
            ln_elementwise_affine=None,
            positional_embedding_base=self.config.rope_theta,
            device=self.device,
            dtype=dtype,
            impl=impl_mode,
            tp_size=tp_size,
            tp_group=tp_group,
        )

    def port_attn_parameter(self):
        self.self_attn.load_parameter(
            q_proj=self.module.self_attn.q_proj,
            k_proj=self.module.self_attn.k_proj,
            v_proj=self.module.self_attn.v_proj,
            out_proj=self.module.self_attn.o_proj,
        )

    def port_mlp_parameter(self):
        # IPEXTransformerMLPOptimizedFp16SiluLlama
        self.mlp.load_parameter(
            self.module.mlp.gate_proj,
            self.module.mlp.down_proj,
            self.module.mlp.up_proj,
        )

    def port_norm_parameter(self):
        self.input_layernorm.weight = self.module.input_layernorm.weight
        self.post_attn_layernorm.weight = self.module.post_attention_layernorm.weight

    def transpose_parameter(self):
        self.self_attn.transpose_parameter()
        self.mlp.transpose_parameter()

    def port_all_parameters_to_new_module(self):
        super().port_all_parameters_to_new_module()
        if self.ipex_config.transpose:
            self.transpose_parameter()
        self.self_attn.cat_qkv()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        # hidden_states:  [bs*beam, seq, hidden_size]
        # position_ids:   [bs*beam, seq]
        # attention_mask: [bs*beam, head, q_seq, kv_seq]
        bs = IPEXTransformerAttn.batch_size
        dim = hidden_states.dim()
        if dim == 3:
            beam = hidden_states.shape[0] // bs
            seq = hidden_states.shape[1]
        elif dim == 4:
            beam = hidden_states.shape[1]
            seq = hidden_states.shape[2]
        else:
            print("Unsupported input shape")
            return

        IPEXTransformerAttn.beam_size = beam

        if self.layer_idx < len(past_key_value):
            if not isinstance(past_key_value, LlamaCache):
                past_key_value = LlamaCache().from_legacy_cache(past_key_value)

            layer_past = (past_key_value.key_cache, past_key_value.value_cache)
        else:
            layer_past = None

        first_token = True if layer_past is None else False

        hidden_size = hidden_states.shape[-1]
        hidden_shape = [bs, beam, seq, hidden_size]
        if first_token and beam > 1:
            # for 1st token, keep the original layout
            # reduce the duplicated info in beam dim
            # shape -> [bs*beam, seq, hidden_size]
            # layout -> [bs*beam, seq, hidden_size]
            hidden_states = hidden_states.view(hidden_shape)[:, 0, :, :].contiguous()
            if position_ids is not None:
                position_ids = position_ids.view(bs, beam, position_ids.shape[1])[
                    :, 0, :
                ].view(bs, position_ids.shape[1])
            if attention_mask is not None:
                attention_mask = attention_mask.view(
                    bs,
                    beam,
                    attention_mask.shape[1],
                    attention_mask.shape[2],
                    attention_mask.shape[3],
                )[:, 0, :, :, :].view(
                    bs,
                    attention_mask.shape[1],
                    attention_mask.shape[2],
                    attention_mask.shape[3],
                )
        else:
            # for 2nd to last token, we convert the layout
            # shape -> [bs*beam, seq, hidden_size]
            # convert layout form [bs*beam, seq, hidden_size] to [seq, bs*beam, hidden_size]
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            output_attentions=output_attentions,
            use_cache=use_cache,
            residual=residual,
            first_token=first_token,
        )
        past_key_value.update(
            present_key_value[0], present_key_value[1], self.layer_idx
        )

        residual = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)
        if first_token and beam > 1:
            # for 1st token, expand the result with beam
            hidden_states = hidden_states.view(bs, 1, seq, hidden_size).expand(
                [bs, beam, seq, hidden_size]
            )
        elif ipex._C._has_2d_block_array(0) or (beam > 1 and not first_token):
            # for 2nd to last token, we convert the layout back
            # convert hidden_states form [seq, beam, hidden_size] back to [beam, seq, hidden_size]
            hidden_states = hidden_states.transpose(0, 1)
        else:
            hidden_states = hidden_states.transpose(0, 1)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (past_key_value,)
        return outputs
