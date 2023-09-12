import torch
from torch import nn
from typing import Optional, Tuple, Union
import math
import re
from ...reference.fusions.mha_fusion import (
    _IPEXRopeRef,
    _IPEXScaleDotProductRef,
)


def _GPTJAttention_forward(
    self,
    hidden_states: torch.FloatTensor,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor, Tuple[torch.Tensor]],
    Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
]:
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
    key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)

    key = self._IPEXROPE(
        key,
        position_ids.contiguous(),
        self.num_attention_heads,
        self.head_dim,
        1,  # neighbor elements
        64,
    )
    query = self._IPEXROPE(
        query,
        position_ids.contiguous(),
        self.num_attention_heads,
        self.head_dim,
        1,
        64,
    )
    if use_cache:
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, True)
        (
            attn_output,
            attn_weights,
            present,
        ) = self._IPEXScaleDotProduct(
            query,
            key,
            value,
            self.scale_attn,
            layer_past,
            head_mask,
            attention_mask,
        )
    else:
        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)
        present = None
        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )
    attn_output = self._merge_heads(
        attn_output, self.num_attention_heads, self.head_dim
    )
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # a, present, (attentions)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _LlamaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    query = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    key = self.k_proj(hidden_states).view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    )
    value = self.v_proj(hidden_states).view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    )
    kv_seq_len = (
        q_len + past_key_value[0].size(-2) if past_key_value is not None else q_len
    )
    key = self._IPEXROPE(
        key,
        position_ids,
        self.num_key_value_heads,
        self.head_dim,
        self.head_dim // 2,
        self.head_dim,
        kv_seq_len,
    )
    query = self._IPEXROPE(
        query,
        position_ids,
        self.num_heads,
        self.head_dim,
        self.head_dim // 2,
        self.head_dim,
        kv_seq_len,
    )

    if use_cache:
        (attn_output, attn_weights, past_key_value) = self._IPEXScaleDotProduct(
            query,
            key,
            value,
            math.sqrt(self.head_dim),
            past_key_value,
            None,
            attention_mask,
        )
    else:
        value_states = value.transpose(1, 2)
        query_states = query.transpose(1, 2)
        key_states = key.transpose(1, 2)
        kv_seq_len = key_states.shape[-2]

        past_key_value = None
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = torch.tensor(attn_weights) + torch.tensor(attention_mask)
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _GPTNeoXAttention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    has_layer_past = layer_past is not None

    qkv = self.query_key_value(hidden_states)

    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)

    query = qkv[..., : self.head_size]
    key = qkv[..., self.head_size : 2 * self.head_size]
    value = qkv[..., 2 * self.head_size :]
    seq_len = key.shape[1]

    if has_layer_past:
        seq_len += layer_past[0].shape[-2]

    key = self._IPEXROPE(
        key,
        position_ids,
        self.num_attention_heads,
        self.head_size,
        self.rotary_ndims // 2,
        self.rotary_ndims,
        seq_len,
    )
    query = self._IPEXROPE(
        query,
        position_ids,
        self.num_attention_heads,
        self.head_size,
        self.rotary_ndims // 2,
        self.rotary_ndims,
        seq_len,
    )

    if not use_cache:
        value = value.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        present = None

        # Compute attention
        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )
    else:
        (attn_output, attn_weights, present) = self._IPEXScaleDotProduct(
            query,
            key,
            value,
            self.norm_factor,
            layer_past,
            head_mask,
            attention_mask,
        )
    attn_output = self._merge_heads(
        attn_output, self.num_attention_heads, self.head_size
    )
    attn_output = self.dense(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def _OPTAttention_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    bsz, tgt_len, _ = hidden_states.size()

    if is_cross_attention and past_key_value is not None:
        key = (
            past_key_value[0]
            .view(bsz, tgt_len, self.num_heads, self.head_dim)
            .contiguous()
        )
        value = (
            past_key_value[1]
            .view(bsz, tgt_len, self.num_heads, self.head_dim)
            .contiguous()
        )
    elif is_cross_attention:
        key = (
            self.k_proj(key_value_states)
            .view(bsz, tgt_len, self.num_heads, self.head_dim)
            .contiguous()
        )
        value = (
            self.v_proj(key_value_states)
            .view(bsz, tgt_len, self.num_heads, self.head_dim)
            .contiguous()
        )
    else:
        key = (
            self.k_proj(hidden_states)
            .view(bsz, tgt_len, self.num_heads, self.head_dim)
            .contiguous()
        )
        value = (
            self.v_proj(hidden_states)
            .view(bsz, tgt_len, self.num_heads, self.head_dim)
            .contiguous()
        )
    query = (
        self.q_proj(hidden_states)
        .view(bsz, tgt_len, self.num_heads, self.head_dim)
        .contiguous()
    )

    (
        attn_output,
        attn_weights,
        past_key_value_decoder,
    ) = self._IPEXScaleDotProduct(
        query,
        key,
        value,
        1 / self.scaling,
        past_key_value,
        layer_head_mask,
        attention_mask,
    )

    if self.is_decoder:
        past_key_value = past_key_value_decoder

    if not output_attentions:
        attn_weights_reshaped = None
    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)

    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned aross GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    return attn_output, attn_weights_reshaped, past_key_value


class _IPEXAttentionRef(nn.Module):
    def __init__(self, module, config, sdp_module_ref, distributed=False):
        super().__init__()
        for k, v in module.__dict__.items():
            setattr(self, k, v)
        for k, v in module.__class__.__dict__.items():
            if k.startswith("__") or k.startswith("forward"):
                continue
            setattr(self.__class__, k, getattr(module.__class__, k))

        self.model_backbone = config.architectures[0]

        # common known as hidden_size
        self.hidden_size = (
            module.hidden_size if hasattr(module, "hidden_size") else module.embed_dim
        )
        # common known as num of attention_heads
        self.num_attention_heads = (
            module.num_attention_heads
            if hasattr(module, "num_attention_heads")
            else module.num_heads
        )
        if hasattr(module, "num_key_value_heads"):
            self.num_key_value_heads = module.num_key_value_heads
        else:
            if hasattr(config, "num_key_value_heads"):
                AssertionError(
                    False,
                    "Your transformers version does not support GQA feature, plese upgrade (>= 4.31.0)",
                )
            else:
                self.num_key_value_heads = self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.head_dim = self.hidden_size // self.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings

        if not re.search("OPT", self.model_backbone, re.IGNORECASE):
            if hasattr(module, "rotary_dim"):
                self.pos_embd_dim = module.rotary_dim
            elif hasattr(module, "rotary_ndims"):
                self.pos_embd_dim = module.rotary_ndims
            else:
                self.pos_embd_dim = self.head_dim
            self.rope_base = (
                config.rotary_emb_base if hasattr(config, "rotary_emb_base") else 10000
            )
            self._IPEXROPE = _IPEXRopeRef(
                self.max_position_embeddings,
                self.pos_embd_dim,
                self.rope_base,
                self.model_backbone,
            )

        self._IPEXScaleDotProduct = _IPEXScaleDotProductRef(module, config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        if re.search("GPTJ", self.model_backbone, re.IGNORECASE):
            return _GPTJAttention_forward(
                self,
                hidden_states,
                layer_past,
                attention_mask,
                position_ids,
                head_mask,
                use_cache,
                output_attentions,
            )
        elif re.search("llama", self.model_backbone, re.IGNORECASE):
            return _LlamaAttention_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        elif re.search("gptneox", self.model_backbone, re.IGNORECASE):
            return _GPTNeoXAttention_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                head_mask,
                layer_past,
                use_cache,
                output_attentions,
            )
        elif re.search("OPT", self.model_backbone, re.IGNORECASE):
            return _OPTAttention_forward(
                self,
                hidden_states,
                key_value_states,
                past_key_value,
                attention_mask,
                layer_head_mask,
                output_attentions,
            )
        else:
            AssertionError(False, "Do not support the optimization of your model yet")


def _reorder_cache(
    self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
) -> Tuple[Tuple[torch.Tensor]]:
    if len(past_key_values[0]) == 4:  # discrete kv_cache
        for layer_past in past_key_values:
            layer_past[3][layer_past[0].size(-2) - 1] = beam_idx
        return past_key_values
    else:
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past_key_values
        )


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full(
        (tgt_len, tgt_len),
        torch.tensor(torch.finfo(dtype).min, device=device),
        device=device,
    )
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
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        ).to(inputs_embeds.device)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else torch.tensor(expanded_attn_mask)
            + torch.tensor(combined_attention_mask)
        )

    return combined_attention_mask
