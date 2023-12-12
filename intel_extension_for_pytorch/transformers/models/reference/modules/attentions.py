import torch
from torch import nn
from typing import Optional, Tuple, Union
import math
from ...reference.fusions.mha_fusion import (
    _IPEXRopeRef,
    _IPEXScaleDotProductRef,
)
from ..fusions.linear_fusion import (
    _IPEXConcatLinearRef,
)
from torch.nn import functional as F


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
    if hasattr(self, "concat_qkv"):
        query, key, value = self.concat_qkv(hidden_states)
    else:
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
            self.scale_attn_value,
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
    if hasattr(self, "concat_qkv"):
        query, key, value = self.concat_qkv(hidden_states)
    else:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
    query = query.view(bsz, q_len, self.num_heads, self.head_dim)
    key = key.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value = value.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
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
            self.norm_factor_value,
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


def _FalconAttention_forward(
    self,
    hidden_states: torch.Tensor,
    alibi: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
):
    fused_qkv = self.query_key_value(
        hidden_states
    )  # [batch_size, seq_length, 3 x hidden_size]
    num_kv_heads = (
        self.num_heads if self.new_decoder_architecture else self.num_kv_heads
    )

    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
    batch_size, query_length, _, _ = query_layer.shape

    past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]

    if self.rotary:
        seq_len = query_length + torch.tensor(past_kv_length)
        key_layer = self._IPEXROPE(
            key_layer,
            torch.tensor(past_kv_length),
            num_kv_heads,
            self.head_dim,
            self.head_dim // 2,
            self.head_dim,
            seq_len,
        )
        query_layer = self._IPEXROPE(
            query_layer,
            torch.tensor(past_kv_length),
            self.num_heads,
            self.head_dim,
            self.head_dim // 2,
            self.head_dim,
            seq_len,
        )

    attention_mask_float = (
        (attention_mask * 1.0)
        .masked_fill(attention_mask, float("-1e9"))
        .to(query_layer.dtype)
    )

    if use_cache:
        (context_layer, attention_scores, present) = self._IPEXScaleDotProduct(
            query_layer,
            key_layer,
            value_layer,
            math.sqrt(self.head_dim),
            layer_past,
            head_mask,
            attention_mask_float
            + alibi.view(batch_size, self.num_heads, 1, -1) * self.inv_norm_factor
            if alibi is not None
            else attention_mask_float,
            alibi,
        )

        context_layer = context_layer.permute(0, 2, 1, 3)
        attn_output = context_layer.reshape(
            batch_size, query_length, self.num_heads * self.head_dim
        )
    else:
        query_layer = query_layer.transpose(1, 2).reshape(
            batch_size * self.num_heads, query_length, self.head_dim
        )
        key_layer = key_layer.transpose(1, 2).reshape(
            batch_size * num_kv_heads,
            query_length,
            self.head_dim,
        )
        value_layer = value_layer.transpose(1, 2).reshape(
            batch_size * num_kv_heads, query_length, self.head_dim
        )

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, kv_length, head_dim]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, kv_length, _ = key_layer.shape
        attention_mask_float = (
            (attention_mask * 1.0)
            .masked_fill(attention_mask, float("-1e9"))
            .to(query_layer.dtype)
        )
        query_layer_ = query_layer.reshape(
            batch_size, self.num_heads, -1, self.head_dim
        )
        key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
        present = None
        if alibi is None:
            if output_attentions:
                # F.scaled_dot_product_attention doesn't return the attention weights, so we have
                # to do it by hand if we want them
                attention_scores = query_layer_ @ key_layer_.transpose(-1, -2)
                attention_scores /= math.sqrt(self.head_dim)
                attention_scores = F.softmax(
                    attention_scores + attention_mask_float,
                    dim=-1,
                    dtype=hidden_states.dtype,
                )
                attn_output = attention_scores @ value_layer_
            else:
                attn_output = F.scaled_dot_product_attention(
                    query_layer_,
                    key_layer_,
                    value_layer_,
                    attention_mask_float,
                    0.0,
                    is_causal=False,
                )
                attention_scores = None
            attn_output = attn_output.view(
                batch_size, self.num_heads, query_length, self.head_dim
            )
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(
                batch_size, query_length, self.num_heads * self.head_dim
            )
        else:
            matmul_result = query_layer_ @ key_layer_.transpose(-1, -2)
            # change view to [batch_size, num_heads, q_length, kv_length]
            attention_scores = matmul_result.view(
                batch_size, self.num_heads, query_length, kv_length
            )
            # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype
            # [batch_size, num_heads, q_length, kv_length]
            input_dtype = attention_scores.dtype
            # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
            if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                attention_scores = attention_scores.to(torch.float32)
            # Matt (HF) note: We could possibly use F.scaled_dot_product_attention here too, by
            # adding (alibi * self.inv_norm_factor) to attention_mask_float. I think this would be mathematically
            # equivalent and more performant, but there might be a numerical difference. If you're reading this
            # and you'd like to experiment and maybe file a PR, feel free!
            attention_logits = attention_scores + alibi.view(
                batch_size, self.num_heads, 1, -1
            )
            attention_logits *= self.inv_norm_factor
            attention_probs = F.softmax(
                attention_logits + attention_mask_float,
                dim=-1,
                dtype=hidden_states.dtype,
            )
            # [batch_size, num_heads, q_length, kv_length]
            attention_probs = self.attention_dropout(attention_probs)
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            # change view [batch_size, num_heads, q_length, kv_length]
            attention_probs_reshaped = attention_probs.view(
                batch_size, self.num_heads, query_length, kv_length
            )
            # matmul: [batch_size * num_heads, q_length, head_dim]
            context_layer = (attention_probs_reshaped @ value_layer_).flatten(0, 1)
            # change view [batch_size, q_length, num_heads * head_dim]
            attn_output = self._merge_heads(context_layer)

    output_tensor = self.dense(attn_output)
    # output_tensor = attn_output
    if output_attentions:
        return (
            output_tensor,
            present,
            attention_scores if alibi is None else attention_probs,
        )
    else:
        return output_tensor, present


def _BloomAttention_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    alibi: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = True,
    output_attentions: bool = False,
):
    fused_qkv = self.query_key_value(
        hidden_states
    )  # [batch_size, seq_length, 3 x hidden_size]

    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    batch_size, q_length, _, _ = query_layer.shape
    query_layer = query_layer.contiguous()
    key_layer = key_layer.contiguous()
    value_layer = value_layer.contiguous()
    new_alibi = (
        alibi.repeat(1, q_length, 1)
        .view(batch_size, self.num_heads, q_length, -1)
        .contiguous()
    )
    (context_layer, attention_probs, present) = self._IPEXScaleDotProduct(
        query_layer,
        key_layer,
        value_layer,
        1 / self.inv_norm_factor,
        layer_past,
        head_mask,
        attention_mask + new_alibi if attention_mask is not None else new_alibi,
        alibi,
    )

    if not use_cache:
        present = None
    # change view [batch_size, q_length, num_heads * head_dim]
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    context_layer = context_layer.view(batch_size, q_length, -1)

    # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
    if self.pretraining_tp > 1 and self.slow_but_exact:
        slices = self.hidden_size / self.pretraining_tp
        output_tensor = torch.zeros_like(context_layer)
        for i in range(self.pretraining_tp):
            output_tensor = output_tensor + F.linear(
                context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
            )
    else:
        output_tensor = self.dense(context_layer)

    output_tensor = output_tensor + residual

    outputs = (output_tensor, present)
    if output_attentions:
        outputs += (attention_probs,)

    return outputs


def _CodeGenAttention_forward(
    self,
    hidden_states: Optional[torch.FloatTensor],
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
    qkv = self.qkv_proj(hidden_states)
    # TODO(enijkamp): factor out number of logical TPU-v4 cores or make forward pass agnostic
    mp_num = 4
    qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

    local_dim = self.head_dim * self.num_attention_heads // mp_num
    query, value, key = torch.split(qkv_split, local_dim, dim=-1)
    query = self._split_heads(
        query, self.num_attention_heads, self.head_dim, mp_num=mp_num
    ).contiguous()
    key = self._split_heads(
        key, self.num_attention_heads, self.head_dim, mp_num=mp_num
    ).contiguous()
    value = self._split_heads(
        value, self.num_attention_heads, self.head_dim, mp_num=mp_num
    ).contiguous()

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
        (
            attn_output,
            attn_weights,
            present,
        ) = self._IPEXScaleDotProduct(
            query,
            key,
            value,
            self.scale_attn_value,
            layer_past,
            head_mask,
            attention_mask,
        )
    else:
        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
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


def _BaichuanAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    output_attentions: bool = False,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    proj = self.W_pack(hidden_states)
    proj = proj.split([self.hidden_size, self.hidden_size, self.hidden_size], dim=-1)

    query = proj[0].view(bsz, q_len, self.num_heads, self.head_dim)
    key = proj[1].view(bsz, q_len, self.num_heads, self.head_dim)
    value = proj[2].view(bsz, q_len, self.num_heads, self.head_dim)
    if attention_mask is not None:
        if len(attention_mask.size()) == 4:
            attention_mask = attention_mask[:, :, -q_len:, :]
        else:
            attention_mask = attention_mask[:, -q_len:, :]

    kv_seq_len = key.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if hasattr(self, "rotary_emb"):
        key = self._IPEXROPE(
            key,
            position_ids,
            self.num_heads,
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

    (
        attn_output,
        attn_weights,
        present,
    ) = self._IPEXScaleDotProduct(
        query,
        key,
        value,
        math.sqrt(self.head_dim),
        past_key_value,
        None,
        attention_mask,
    )

    if not use_cache:
        present = None
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    # attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, present


def _GLM2Attention_forward(
    self,
    hidden_states,
    attention_mask,
    rotary_pos_emb,
    kv_cache=None,
    use_cache=True,
):
    # hidden_states: [sq, b, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    # =====================
    # Query, Key, and Value
    # =====================

    # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
    mixed_x_layer = self.query_key_value(hidden_states)
    mixed_x_layer = mixed_x_layer.transpose(0, 1)

    if self.multi_query_attention:
        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [
                self.num_attention_heads_per_partition
                * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition
                * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition
                * self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )
        query_layer = query_layer.view(
            query_layer.size()[:-1]
            + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
        )
        key_layer = key_layer.view(
            key_layer.size()[:-1]
            + (
                self.num_multi_query_groups_per_partition,
                self.hidden_size_per_attention_head,
            )
        )
        value_layer = value_layer.view(
            value_layer.size()[:-1]
            + (
                self.num_multi_query_groups_per_partition,
                self.hidden_size_per_attention_head,
            )
        )
    else:
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = self.split_tensor_along_last_dim(
            mixed_x_layer, 3
        )
    past_len = kv_cache[0].shape[-2] if kv_cache is not None else 0
    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        key_layer = self._IPEXROPE(
            key_layer,
            torch.tensor(past_len),
            key_layer.size(-2),
            key_layer.size(-1),
            1,
            64,
        )
        query_layer = self._IPEXROPE(
            query_layer,
            torch.tensor(past_len),
            query_layer.size(-2),
            query_layer.size(-1),
            1,
            64,
        )

    if attention_mask is None:
        attention_mask = torch.ones(
            query_layer.size(0),
            1,
            past_len + query_layer.size(1),
            past_len + key_layer.size(1),
            dtype=torch.bool,
        )
        attention_mask.tril_()
        attention_mask = ~attention_mask
    (
        attn_output,
        attn_weights,
        present,
    ) = self._IPEXScaleDotProduct(
        query_layer,
        key_layer,
        value_layer,
        self.factor,
        kv_cache,
        None,
        attention_mask,
    )
    context_layer = attn_output.permute(2, 0, 1, 3).contiguous()
    output = context_layer.reshape(
        context_layer.shape[0], context_layer.shape[1], self.projection_size
    )
    # output = self.dense(context_layer)
    return output, present


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
        if hasattr(module, "hidden_size"):
            self.hidden_size = module.hidden_size
        elif hasattr(module, "embed_dim"):
            self.hidden_size = module.embed_dim
        elif hasattr(module, "hidden_size_per_attention_head"):
            self.hidden_size = module.hidden_size_per_attention_head

        # common known as num of attention_heads
        if hasattr(module, "num_attention_heads"):
            self.num_attention_heads = module.num_attention_heads
        elif hasattr(module, "num_heads"):
            self.num_attention_heads = module.num_heads
        elif hasattr(module, "num_attention_heads_per_partition"):
            self.num_attention_heads = module.num_attention_heads_per_partition

        if hasattr(module, "num_key_value_heads"):
            self.num_key_value_heads = module.num_key_value_heads
        else:
            if hasattr(config, "num_key_value_heads"):
                raise ValueError(
                    "Your transformers version does not support GQA feature, plese upgrade (>= 4.31.0)"
                )
            else:
                self.num_key_value_heads = self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.head_dim = self.hidden_size // self.num_attention_heads

        self.max_position_embeddings = (
            config.max_position_embeddings
            if hasattr(config, "max_position_embeddings")
            else 2048
        )

        if (
            self.model_backbone
            not in [
                "OPTForCausalLM",
                "BloomForCausalLM",
            ]
            or self.model_backbone == "BaichuanForCausalLM"
            and hasattr(module, "rotary_emb")
        ):
            if hasattr(module, "rotary_dim"):
                self.pos_embd_dim = module.rotary_dim
            elif hasattr(module, "rotary_ndims"):
                self.pos_embd_dim = module.rotary_ndims
            elif self.model_backbone == "ChatGLMModel":
                rotary_dim = (
                    config.hidden_size // config.num_attention_heads
                    if config.kv_channels is None
                    else config.kv_channels
                )
                self.pos_embd_dim = rotary_dim // 2
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

        if (
            self.model_backbone == "GPTJForCausalLM"
            or self.model_backbone == "LlamaForCausalLM"
        ):
            if (
                hasattr(module, "q_proj")
                and hasattr(module, "k_proj")
                and hasattr(module, "v_proj")
            ):

                def get_weight_shape(mod):
                    if hasattr(mod, "in_features") and hasattr(mod, "out_features"):
                        return [mod.in_features, mod.out_features]
                    elif hasattr(mod, "weight") and hasattr(mod.weight, "shape"):
                        return list(mod.weight.shape)
                    return None

                weight_shapes = [
                    get_weight_shape(mod)
                    for mod in [module.q_proj, module.k_proj, module.v_proj]
                ]
                if weight_shapes[0] is not None and all(
                    weight_shapes[0] == shape for shape in weight_shapes[1:]
                ):
                    self.concat_qkv = _IPEXConcatLinearRef(
                        [module.q_proj, module.k_proj, module.v_proj]
                    )
                    del module.q_proj, module.k_proj, module.v_proj

        self._IPEXScaleDotProduct = _IPEXScaleDotProductRef(module, config)

        if (
            self.model_backbone == "FalconForCausalLM"
            or self.model_backbone == "RWForCausalLM"
        ):
            self.split_size = self.hidden_size
            self.hidden_dropout = config.hidden_dropout
            self.rotary = config.rotary

            # Layer-wise attention scaling
            self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
            self.beta = self.inv_norm_factor
            self.multi_query = (
                config.multi_query if hasattr(config, "multi_query") else None
            )
            if hasattr(config, "new_decoder_architecture"):
                is_new_decoder_architecture = config.new_decoder_architecture
            else:
                is_new_decoder_architecture = hasattr(
                    config, "num_kv_heads"
                ) or hasattr(config, "n_head_kv")
            if is_new_decoder_architecture or not self.multi_query:
                num_kv_heads = (
                    config.num_kv_heads
                    if hasattr(config, "num_kv_heads")
                    else config.n_head_kv
                )
                if num_kv_heads == config.num_attention_heads:
                    self.num_kv_heads = self.num_heads
                else:
                    if hasattr(module, "num_kv_heads"):
                        if module.num_kv_heads != num_kv_heads:
                            self.num_kv_heads = module.num_kv_heads
                        else:
                            self.num_kv_heads = num_kv_heads // (
                                config.num_attention_heads // module.num_heads
                            )
                    elif hasattr(module, "num_kv"):
                        if module.num_kv != num_kv_heads:
                            self.num_kv_heads = module.num_kv
                        else:
                            self.num_kv_heads = num_kv_heads // (
                                config.num_attention_heads // module.num_heads
                            )
            else:
                self.num_kv_heads = 1
            self.new_decoder_architecture = is_new_decoder_architecture
        elif self.model_backbone == "ChatGLMModel":
            self.projection_size = module.projection_size
            # Per attention head and per partition values.
            self.hidden_size_per_attention_head = module.hidden_size_per_attention_head
            self.num_attention_heads_per_partition = (
                module.num_attention_heads_per_partition
            )
            self.multi_query_attention = module.multi_query_attention
            self.qkv_hidden_size = module.qkv_hidden_size
            if self.multi_query_attention:
                self.num_multi_query_groups_per_partition = (
                    module.num_multi_query_groups_per_partition
                )
                self.qkv_hidden_size = module.qkv_hidden_size
            self.query_key_value = module.query_key_value

            self.core_attention = module.core_attention
            self.dense = module.dense
            self.factor = (
                self.core_attention.norm_factor
                if self.core_attention.coeff is None
                else self.core_attention.norm_factor / self.core_attention.coeff
            )
            self.text_max_length = (
                config.text_max_length if hasattr(config, "text_max_length") else 2048
            )
            self.factor = (
                module.core_attention.norm_factor
                if module.core_attention.coeff is None
                else module.core_attention.norm_factor / module.core_attention.coeff
            )
            self.layer_number = module.layer_number
        if self.model_backbone in ["GPTJForCausalLM", "CodeGenForCausalLM"]:
            self.scale_attn_value = self.scale_attn.item()
        if self.model_backbone == "GPTNeoXForCausalLM":
            self.norm_factor_value = self.norm_factor.item()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        key_value_states: Optional[torch.Tensor] = None,
        kv_caches: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        alibi: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ):
        if self.model_backbone == "GPTJForCausalLM":
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
        elif self.model_backbone == "LlamaForCausalLM":
            return _LlamaAttention_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        elif self.model_backbone == "GPTNeoXForCausalLM":
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
        elif self.model_backbone == "OPTForCausalLM":
            return _OPTAttention_forward(
                self,
                hidden_states,
                key_value_states,
                past_key_value,
                attention_mask,
                layer_head_mask,
                output_attentions,
            )
        elif (
            self.model_backbone == "FalconForCausalLM"
            or self.model_backbone == "RWForCausalLM"
        ):
            return _FalconAttention_forward(
                self,
                hidden_states,
                alibi,
                attention_mask,
                layer_past,
                head_mask,
                use_cache,
                output_attentions,
            )
        elif self.model_backbone == "BloomForCausalLM":
            return _BloomAttention_forward(
                self,
                hidden_states,
                residual,
                alibi,
                attention_mask,
                layer_past,
            )
        elif self.model_backbone == "CodeGenForCausalLM":
            return _CodeGenAttention_forward(
                self,
                hidden_states,
                layer_past,
                attention_mask,
                position_ids,
                head_mask,
                use_cache,
                output_attentions,
            )
        elif self.model_backbone == "BaichuanForCausalLM":
            return _BaichuanAttention_forward(
                self,
                hidden_states,
                attention_mask,
                past_key_value,
                position_ids,
                output_attentions,
                use_cache,
            )
        elif self.model_backbone == "ChatGLMModel":
            return _GLM2Attention_forward(
                self,
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_caches,
                use_cache,
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


def _convert_cache_to_standard_format(
    self, past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    if len(past_key_value[0]) == 4:
        return past_key_value

    batch_size_times_num_heads, kv_length, head_dim = past_key_value[0][0].shape

    num_heads = batch_size_times_num_heads // batch_size
    return tuple(
        (
            layer_past[0].view(batch_size, num_heads, kv_length, head_dim),
            layer_past[1].view(batch_size, num_heads, kv_length, head_dim),
        )
        for layer_past in past_key_value
    )


def _convert_to_rw_cache(
    self, past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    if len(past_key_value[0]) == 4:
        return past_key_value
    batch_size, num_heads, kv_length, head_dim = past_key_value[0][0].shape
    batch_size_times_num_heads = batch_size * num_heads
    return tuple(
        (
            layer_past[0].view(batch_size_times_num_heads, kv_length, head_dim),
            layer_past[1].view(batch_size_times_num_heads, kv_length, head_dim),
        )
        for layer_past in past_key_value
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


def _make_causal_mask_falcon(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention. This mask does not take the existing attention mask into account - it
    just blocks tokens from attending forwards in the sequence. The output shape will be `[batch_size, 1,
    target_length, target_length+past_key_values_length]`.
    """
    batch_size, target_length = input_ids_shape

    mask = torch.triu(
        torch.ones((target_length, target_length), dtype=torch.bool, device=device),
        diagonal=1,
    )
    # If past_key_values_length is 0 this is an empty tensor and the concatenation is a no-op.
    # This code style is an unfortunate consequence of getting your TF engineer to port models; doing it this
    # way avoids a data-dependent conditional, which will help me when I have to port this to XLA later.
    past_mask = torch.zeros(
        (target_length, past_key_values_length), dtype=torch.bool, device=device
    )
    mask = torch.cat([past_mask, mask], dim=-1)
    expanded_mask = mask[None, None, :, :].expand(
        batch_size, 1, target_length, target_length + past_key_values_length
    )
    return expanded_mask


def _expand_mask_falcon(
    mask: torch.Tensor, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, seq_length]` to `[batch_size, 1, seq_length, seq_length + past_length]`.
    """
    batch_size, total_length = mask.shape
    seq_length = (
        total_length - past_key_values_length
        if past_key_values_length is not None
        else total_length
    )

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, seq_length, total_length)


def _prepare_attn_mask_falcon(
    self,
    attention_mask: torch.Tensor,
    input_shape: Tuple[int, int],
    past_key_values_length: int,
) -> torch.BoolTensor:
    # Create a causal mask
    # The attention mask we receive as input should cover the whole extended sequence, including any past
    # cache, so its shape should be [batch_size, seq_length + past_key_values_length]
    # The output shape will be [batch_size, 1, seq_length, seq_length + past_key_values_length]
    if input_shape[1] + past_key_values_length != attention_mask.shape[1]:
        raise ValueError(
            "Attention mask shape should be (batch_size, seq_length + past_key_values_length)"
            f" but is {attention_mask.shape} with input_ids shape {input_shape} and past length"
            f" {past_key_values_length}."
        )
    combined_attention_mask = None
    device = attention_mask.device
    _, seq_length = input_shape

    if seq_length > 1:
        combined_attention_mask = _make_causal_mask_falcon(
            input_shape, device=device, past_key_values_length=past_key_values_length
        )

    # [batch_size, seq_length + past_key_values_length] -> [batch_size, 1, seq_length, seq_length + past_key_values_length]
    expanded_attn_mask = _expand_mask_falcon(
        attention_mask, past_key_values_length=past_key_values_length
    )
    combined_attention_mask = (
        expanded_attn_mask
        if combined_attention_mask is None
        else expanded_attn_mask | combined_attention_mask
    )

    return combined_attention_mask


def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def _gen_baichuan_alibi_mask(n_head, max_pos):
    """used in inference only"""
    slopes = torch.Tensor(_get_interleave(n_head))
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_pos).unsqueeze(
        0
    ).unsqueeze(0).expand(n_head, -1, -1)
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(
        torch.zeros([max_pos, max_pos]).float().fill_(float("-inf")).type_as(alibi), 1
    )
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask


def GLM2_get_masks(self, input_ids, past_key_values, padding_mask=None):
    batch_size, seq_length = input_ids.shape
    full_attention_mask = torch.ones(
        batch_size, seq_length, seq_length, device=input_ids.device
    )
    full_attention_mask.tril_()
    past_length = 0
    if past_key_values:
        if len(past_key_values[0]) != 4:  # not discrete kv cache
            past_length = past_key_values[0][0].shape[0]
        else:  # discrete kv cache
            past_length = past_key_values[0][0].shape[-2]
    if past_length:
        full_attention_mask = torch.cat(
            (
                torch.ones(
                    batch_size, seq_length, past_length, device=input_ids.device
                ),
                full_attention_mask,
            ),
            dim=-1,
        )
    if padding_mask is not None:
        full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
    # if not past_length and padding_mask is not None:
    #     full_attention_mask -= padding_mask.unsqueeze(-1) - 1
    full_attention_mask = (full_attention_mask < 0.5).bool()
    full_attention_mask.unsqueeze_(1)
    return full_attention_mask
