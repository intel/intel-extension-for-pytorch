import torch
from torch import nn
from typing import Optional, Tuple, Union
import math
from transformers.models.llama.configuration_llama import LlamaConfig
from torch.nn import functional as F
from transformers.models.bloom.modeling_bloom import dropout_add
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import (
    upcast_softmax,
    upcast_masked_softmax,
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
            past_length = past_key_values[0][3]
    full_attention_mask = torch.cat(
        (
            torch.ones(batch_size, seq_length, past_length, device=input_ids.device),
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


def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum(
        "i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq
    ).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


def create_embed_positions_for_falcon(
    seq_len: int, head_dim: int, past_key_values_length
) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim))
    freqs = torch.einsum(
        "i , j -> i j",
        torch.arange(seq_len + past_key_values_length, dtype=torch.float),
        inv_freq,
    ).float()
    return torch.cat((freqs.sin().repeat(1, 2), freqs.cos().repeat(1, 2)), dim=-1)


class _GPTJAttention(nn.Module):
    def __init__(self, module, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = module.embed_dim
        self.num_attention_heads = module.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        ).to(torch.get_default_dtype())

        self.enable_concat_linear = getattr(config, "weight_only_quantization", False)
        self.k_proj = module.k_proj
        self.v_proj = module.v_proj
        self.q_proj = module.q_proj

        if self.enable_concat_linear:
            weights = torch.cat(
                [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0
            )
            if self.q_proj.bias is not None:
                biases = torch.cat(
                    [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0
                )
                self.concat_qkv = torch.nn.Linear(
                    weights.shape[0], weights.shape[1], bias=True
                )
                self.concat_qkv.bias = torch.nn.Parameter(biases)
            else:
                self.concat_qkv = torch.nn.Linear(
                    weights.shape[0], weights.shape[1], bias=False
                )
            self.concat_qkv.weight = torch.nn.Parameter(weights)
            self.concat_qkv._num_concats = 3

        self.out_proj = module.out_proj
        self.rotary_dim = module.rotary_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)
        self.text_max_length = (
            config.text_max_length if hasattr(config, "text_max_length") else 2048
        )

    def _split_heads(self, tensor, num_attention_heads, attn_head_size, rotary):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        if rotary:
            return tensor
        if len(tensor.shape) == 5:
            return tensor.permute(
                0, 1, 3, 2, 4
            )  # (batch, blocks, head, block_length, head_features)
        elif len(tensor.shape) == 4:
            return tensor.permute(
                0, 2, 1, 3
            )  # (batch, head, seq_length, head_features)
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}"
            )

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}"
            )
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        attn_weights = attn_weights / self.scale_attn

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _get_embed_positions(self, position_ids):
        embed_positions = self.embed_positions
        if embed_positions.device != position_ids.device:
            embed_positions = embed_positions.to(position_ids.device)
            self.embed_positions = embed_positions
        return embed_positions.repeat(position_ids.shape[0], 1, 1)

    def forward(
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
        if self.enable_concat_linear:
            num_concats = self.concat_qkv._num_concats
            assert num_concats == 3
            qkv_output = self.concat_qkv(hidden_states)
            hidden_size = qkv_output.shape[-1] // num_concats
            qkv = qkv_output.view(num_concats, -1, hidden_size)
            expected_shape = list(hidden_states.shape)[:-1] + [hidden_size]
            query, key, value = (
                qkv[0].view(expected_shape),
                qkv[1].view(expected_shape),
                qkv[2].view(expected_shape),
            )
        else:
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)

        position_ids = position_ids.contiguous()
        torch.ops.torch_ipex.rotary_position_embedding(
            key,
            self.embed_positions,
            position_ids,
            self.num_attention_heads,
            self.head_dim,
            1,  # neighbor elements
            64,
        )
        torch.ops.torch_ipex.rotary_position_embedding(
            query,
            self.embed_positions,
            position_ids,
            self.num_attention_heads,
            self.head_dim,
            1,
            64,
        )
        if use_cache:
            value = self._split_heads(
                value, self.num_attention_heads, self.head_dim, True
            )
            key_cache = layer_past[0].contiguous()
            value_cache = layer_past[1].contiguous()
            beam_idx = layer_past[2].contiguous()
            decoded_tokens = layer_past[3].contiguous()[0]
            (
                attn_output,
                attn_weights,
                key_cache,
                value_cache,
                beam_idx,
            ) = torch.ops.torch_ipex.masked_multihead_self_attention(
                query,
                key,
                value,
                key_cache,
                value_cache,
                beam_idx,
                decoded_tokens,
                self.scale_attn,
                self.text_max_length,
                head_mask,
                attention_mask,
            )
            present = (
                key_cache,
                value_cache,
                beam_idx,
                torch.tensor(layer_past[3] + query.shape[1], dtype=torch.long),
            )
        else:
            key = key.permute(0, 2, 1, 3)
            query = query.permute(0, 2, 1, 3)
            value = self._split_heads(
                value, self.num_attention_heads, self.head_dim, False
            )
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


def _LlamaRMSNorm_forward_v1(self, hidden_states):
    return torch.ops.torch_ipex.rmsnorm(
        hidden_states, self.weight, self.variance_epsilon
    )


def _LlamaRMSNorm_forward_v2(self, hidden_states):
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

    # convert into half-precision if necessary
    if self.weight.dtype in [torch.float16, torch.bfloat16]:
        hidden_states = hidden_states.to(self.weight.dtype)

    return self.weight * hidden_states


class _LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, module, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = module.hidden_size
        self.num_heads = module.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = self.config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = module.q_proj
        self.k_proj = module.k_proj
        self.v_proj = module.v_proj
        self.o_proj = module.o_proj
        self.rotary_emb = create_sinusoidal_positions(
            self.max_position_embeddings, self.head_dim
        )
        self.text_max_length = (
            self.config.text_max_length
            if hasattr(self.config, "text_max_length")
            else 2048
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        position_ids = position_ids.contiguous()
        key = key.contiguous()
        query = query.contiguous()
        torch.ops.torch_ipex.rotary_position_embedding(
            key,
            self.rotary_emb,
            position_ids,
            self.num_heads,
            self.head_dim,
            self.head_dim // 2,
            self.head_dim,
        )
        torch.ops.torch_ipex.rotary_position_embedding(
            query,
            self.rotary_emb,
            position_ids,
            self.num_heads,
            self.head_dim,
            self.head_dim // 2,
            self.head_dim,
        )

        if use_cache:
            if past_key_value is None:
                beam_idx_tmp = torch.zeros(
                    (2048, int(4)), dtype=torch.long
                ).contiguous()
                past_key_value = (
                    torch.zeros([1, 32, 1, 128]).contiguous(),
                    torch.zeros([1, 32, 1, 128]).contiguous(),
                    beam_idx_tmp,
                    torch.zeros(1, dtype=torch.long).contiguous(),
                )
            key_cache = past_key_value[0].contiguous()
            value_cache = past_key_value[1].contiguous()
            beam_idx = past_key_value[2].contiguous()
            print("#########key_cache", key_cache.dtype)
            decoded_tokens = past_key_value[3].contiguous()[0]
            (
                attn_output,
                attn_weights,
                key_cache,
                value_cache,
                beam_idx,
            ) = torch.ops.torch_ipex.masked_multihead_self_attention(
                query,
                key,
                value,
                key_cache,
                value_cache,
                beam_idx,
                decoded_tokens,
                math.sqrt(self.head_dim),
                self.text_max_length,
                None,
                attention_mask,
            )
            past_key_value = (
                key_cache,
                value_cache,
                beam_idx,
                torch.tensor(past_key_value[3] + query.shape[1], dtype=torch.long),
            )
        else:
            value_states = value.transpose(1, 2)
            query_states = query.transpose(1, 2)
            key_states = key.transpose(1, 2)
            kv_seq_len = key_states.shape[-2]

            past_key_value = None

            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = torch.tensor(attn_weights) + torch.tensor(attention_mask)
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class _LlamaAttention_GQA(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, module, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = module.hidden_size
        self.num_heads = module.num_heads
        if self.config.num_key_value_heads == self.config.num_attention_heads:
            self.num_key_value_heads = self.num_heads
        else:
            if hasattr(module, "num_key_value_heads"):
                if module.num_key_value_heads != self.config.num_key_value_heads:
                    self.num_key_value_heads = module.num_key_value_heads
                else:  # workaround here as deepspeed does not support llama2 GQA autoTP, will remove once it supports
                    self.num_key_value_heads = self.config.num_key_value_heads // (
                        self.config.num_attention_heads // module.num_heads
                    )
                    if self.num_key_value_heads < 1:
                        AssertionError(
                            "Does not support Tensor parallel in this num_key_value_heads < 1 case, \
                                please reach out deepspeed's support"
                        )

            else:
                AssertionError(
                    "Your transformers version does not support LLaMA2 GQA feature, plese upgrade"
                )

        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = module.q_proj
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.k_proj.weight = module.k_proj.weight
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj.weight = module.v_proj.weight
        self.o_proj = module.o_proj
        self.rotary_emb = create_sinusoidal_positions(
            self.max_position_embeddings, self.head_dim
        )
        self.text_max_length = (
            self.config.text_max_length
            if hasattr(self.config, "text_max_length")
            else 2048
        )

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
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

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        position_ids = position_ids.contiguous()
        key = key.contiguous()
        query = query.contiguous()
        torch.ops.torch_ipex.rotary_position_embedding(
            key,
            self.rotary_emb,
            position_ids,
            self.num_key_value_heads,
            self.head_dim,
            self.head_dim // 2,
            self.head_dim,
        )
        torch.ops.torch_ipex.rotary_position_embedding(
            query,
            self.rotary_emb,
            position_ids,
            self.num_heads,
            self.head_dim,
            self.head_dim // 2,
            self.head_dim,
        )

        if use_cache:
            if past_key_value is None:
                beam_idx_tmp = torch.zeros(
                    (2048, int(4)), dtype=torch.long
                ).contiguous()
                past_key_value = (
                    torch.zeros([1, 32, 1, 128]).contiguous(),
                    torch.zeros([1, 32, 1, 128]).contiguous(),
                    beam_idx_tmp,
                    torch.zeros(1, dtype=torch.long).contiguous(),
                )
            key_cache = past_key_value[0].contiguous()
            value_cache = past_key_value[1].contiguous()
            beam_idx = past_key_value[2].contiguous()
            decoded_tokens = past_key_value[3].contiguous()[0]
            (
                attn_output,
                attn_weights,
                key_cache,
                value_cache,
                beam_idx,
            ) = torch.ops.torch_ipex.masked_multihead_self_attention(
                query,
                key,
                value,
                key_cache,
                value_cache,
                beam_idx,
                decoded_tokens,
                math.sqrt(self.head_dim),
                self.text_max_length,
                None,
                attention_mask,
            )
            past_key_value = (
                key_cache,
                value_cache,
                beam_idx,
                torch.tensor(past_key_value[3] + query.shape[1], dtype=torch.long),
            )
        else:
            value_states = value.transpose(1, 2)
            query_states = query.transpose(1, 2)
            key_states = key.transpose(1, 2)
            kv_seq_len = key_states.shape[-2]

            past_key_value = None
            # repeat k/v heads if n_kv_heads < n_heads
            key_states = self.repeat_kv(key_states, self.num_key_value_groups)
            value_states = self.repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = torch.tensor(attn_weights) + torch.tensor(attention_mask)
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]
        self.sin_cos = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=1)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
            self.sin_cos = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=1)
        return self.sin_cos


class _GPTNeoXAttention(nn.Module):
    def __init__(self, module, config):
        super().__init__()
        self.num_attention_heads = module.num_attention_heads
        self.hidden_size = module.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            config.max_position_embeddings,
            base=config.rotary_emb_base,
        )
        self.norm_factor = torch.sqrt(
            torch.tensor(self.head_size, dtype=torch.float32)
        ).to(torch.get_default_dtype())
        self.query_key_value = module.query_key_value
        self.dense = module.dense
        self.text_max_length = (
            config.text_max_length if hasattr(config, "text_max_length") else 2048
        )

    def _split_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(
            tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size
        )
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]

        query = query.view(
            batch_size * num_attention_heads, query_length, attn_head_size
        )
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(
                torch.tensor(
                    1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device
                )
                / self.norm_factor
            ),
        )
        attn_scores = attn_scores.view(
            batch_size, num_attention_heads, query_length, key_length
        )

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(
            attn_scores.device
        )
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(
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

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size]
        key = qkv[..., self.head_size : 2 * self.head_size]
        value = qkv[..., 2 * self.head_size :]
        seq_len = key.shape[1]
        if has_layer_past:
            seq_len += layer_past[3].item()
        sin_cos = self.rotary_emb(value, seq_len=seq_len)
        position_ids = position_ids.contiguous()
        key = key.contiguous()
        query = query.contiguous()
        torch.ops.torch_ipex.rotary_position_embedding(
            key,
            sin_cos,
            position_ids,
            self.num_attention_heads,
            self.head_size,
            self.rotary_ndims // 2,
            self.rotary_ndims,
        )
        torch.ops.torch_ipex.rotary_position_embedding(
            query,
            sin_cos,
            position_ids,
            self.num_attention_heads,
            self.head_size,
            self.rotary_ndims // 2,
            self.rotary_ndims,
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
            if layer_past is None:
                layer_past = (
                    torch.randn(1),
                    torch.randn(1),
                    torch.zeros(2048, 4, dtype=torch.long),
                    torch.zeros(1, dtype=torch.long),
                )
            key_cache = layer_past[0].contiguous()
            value_cache = layer_past[1].contiguous()
            beam_idx = layer_past[2].contiguous()
            decoded_tokens = layer_past[3].contiguous()[0]
            (
                attn_output,
                attn_weights,
                key_cache,
                value_cache,
                beam_idx,
            ) = torch.ops.torch_ipex.masked_multihead_self_attention(
                query,
                key,
                value,
                key_cache,
                value_cache,
                beam_idx,
                decoded_tokens,
                self.norm_factor,
                self.text_max_length,
                head_mask,
                attention_mask,
            )
            present = (
                key_cache,
                value_cache,
                beam_idx,
                torch.tensor(layer_past[3] + query.shape[1], dtype=torch.long),
            )
        attn_output = self._merge_heads(
            attn_output, self.num_attention_heads, self.head_size
        )
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class _OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, module, config):
        super().__init__()
        self.embed_dim = module.embed_dim
        self.num_heads = module.num_heads
        self.dropout = module.dropout
        self.head_dim = self.embed_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = module.is_decoder

        self.k_proj = module.k_proj
        self.v_proj = module.v_proj
        self.q_proj = module.q_proj
        self.out_proj = module.out_proj
        self.text_max_length = (
            config.text_max_length if hasattr(config, "text_max_length") else 2048
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
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
        if past_key_value is None:
            past_key_value = (
                torch.randn(0),
                torch.randn(0),
                torch.zeros(2048, 4, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )
        query = (
            self.q_proj(hidden_states)
            .view(bsz, tgt_len, self.num_heads, self.head_dim)
            .contiguous()
        )
        key_cache = past_key_value[0].contiguous()
        value_cache = past_key_value[1].contiguous()
        beam_idx = past_key_value[2].contiguous()
        decoded_tokens = past_key_value[3].contiguous()[0]
        (
            attn_output,
            attn_weights,
            key_cache,
            value_cache,
            beam_idx,
        ) = torch.ops.torch_ipex.masked_multihead_self_attention(
            query,
            key,
            value,
            key_cache,
            value_cache,
            beam_idx,
            decoded_tokens,
            1 / self.scaling,
            self.text_max_length,
            layer_head_mask,
            attention_mask,
        )

        if self.is_decoder:
            past_key_value = (
                key_cache,
                value_cache,
                beam_idx,
                torch.tensor(past_key_value[3] + query.shape[1], dtype=torch.long),
            )

        if not output_attentions:
            attn_weights_reshaped = None
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value


class _FalconAttention(nn.Module):
    def __init__(self, module, config):
        super().__init__()

        self.hidden_size = module.hidden_size
        self.num_heads = module.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

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
            is_new_decoder_architecture = hasattr(config, "num_kv_heads") or hasattr(
                config, "n_head_kv"
            )
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
            self.query_key_value = torch.nn.Linear(
                self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.bias
            )
            self.query_key_value.weight = module.query_key_value.weight
            if config.bias:
                self.query_key_value.bias = module.query_key_value.bias
        else:
            self.num_kv_heads = 1
            self.query_key_value = module.query_key_value
        if is_new_decoder_architecture:
            qkv_out_dim = (num_kv_heads * 2 + self.num_heads) * self.head_dim
        elif self.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
        self.new_decoder_architecture = is_new_decoder_architecture
        self.dense = module.dense
        self.attention_dropout = module.attention_dropout
        self.text_max_length = (
            config.text_max_length if hasattr(config, "text_max_length") else 2048
        )

    def _split_heads(
        self, fused_qkv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        batch_size, seq_length, _ = fused_qkv.shape
        if self.new_decoder_architecture:
            qkv = fused_qkv.view(
                batch_size,
                seq_length,
                -1,
                self.num_heads // self.num_kv_heads + 2,
                self.head_dim,
            )
            query = qkv[:, :, :, :-2]
            key = qkv[:, :, :, [-2]]
            value = qkv[:, :, :, [-1]]
            key = torch.broadcast_to(key, query.shape)
            value = torch.broadcast_to(value, query.shape)
            query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
            return query, key, value
        elif not self.multi_query:
            fused_qkv = fused_qkv.view(
                batch_size, seq_length, self.num_heads, 3, self.head_dim
            )
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        else:
            fused_qkv = fused_qkv.view(
                batch_size, seq_length, self.num_heads + 2, self.head_dim
            )
            return (
                fused_qkv[..., :-2, :],
                fused_qkv[..., [-2], :],
                fused_qkv[..., [-1], :],
            )

    # Copied from transformers.models.bloom.modeling_bloom.BloomAttention._merge_heads
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
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
        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        past_kv_length = 0
        if layer_past is not None:
            if len(layer_past) != 4:
                past_kv_length = layer_past[0].shape[1]
            else:
                past_kv_length = layer_past[3][0]
        if self.rotary:
            embed_positions = create_embed_positions_for_falcon(
                query_length, self.head_dim, torch.tensor(past_kv_length).contiguous()
            )
            torch.ops.torch_ipex.rotary_position_embedding(
                key_layer,
                embed_positions,
                torch.tensor(past_kv_length).contiguous(),
                num_kv_heads,
                self.head_dim,
                self.head_dim // 2,
                self.head_dim,
            )
            torch.ops.torch_ipex.rotary_position_embedding(
                query_layer,
                embed_positions,
                torch.tensor(past_kv_length).contiguous(),
                self.num_heads,
                self.head_dim,
                self.head_dim // 2,
                self.head_dim,
            )
        attention_mask_float = (
            (attention_mask * 1.0)
            .masked_fill(attention_mask, float("-1e9"))
            .to(query_layer.dtype)
        )
        if use_cache:
            if layer_past is None:
                layer_past = (
                    torch.randn(0),
                    torch.randn(0),
                    torch.zeros(2048, 4, dtype=torch.long),
                    torch.zeros(1, dtype=torch.long),
                )
            query = query_layer.contiguous()
            key = key_layer.contiguous()
            value = value_layer.contiguous()
            key_cache = layer_past[0].contiguous()
            value_cache = layer_past[1].contiguous()
            beam_idx = layer_past[2].contiguous()
            decoded_tokens = layer_past[3].contiguous()[0]
            (
                context_layer,
                attn_weights,
                key_cache,
                value_cache,
                beam_idx,
            ) = torch.ops.torch_ipex.masked_multihead_self_attention(
                query,
                key,
                value,
                key_cache,
                value_cache,
                beam_idx,
                decoded_tokens,
                math.sqrt(self.head_dim),
                self.text_max_length,
                head_mask,
                attention_mask_float
                + alibi.view(batch_size, self.num_heads, 1, -1) * self.inv_norm_factor
                if alibi is not None
                else attention_mask_float,
            )
            present = (
                key_cache,
                value_cache,
                beam_idx,
                torch.tensor(layer_past[3] + query.shape[1], dtype=torch.long),
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
            value_layer_ = value_layer.reshape(
                batch_size, num_kv_heads, -1, self.head_dim
            )
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
            return output_tensor, present, attention_probs
        else:
            return output_tensor, present


class _BloomAttention(nn.Module):
    def __init__(self, module, config):
        super().__init__()

        self.pretraining_tp = module.pretraining_tp
        self.slow_but_exact = module.slow_but_exact

        self.hidden_size = module.hidden_size
        self.num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.split_size = module.split_size
        self.hidden_dropout = module.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.query_key_value = module.query_key_value
        self.dense = module.dense
        self.attention_dropout = module.attention_dropout
        self.text_max_length = (
            config.text_max_length if hasattr(config, "text_max_length") else 2048
        )

    def _split_heads(
        self, fused_qkv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
        storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(
            batch_size, seq_length, self.num_heads, 3, self.head_dim
        )
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`, *required*): [batch_size, num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        batch_size, num_heads, seq_length, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
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
        if layer_past is None:
            layer_past = (
                torch.randn(0),
                torch.randn(0),
                torch.zeros(2048, 4, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )
        key_cache = layer_past[0].contiguous()
        value_cache = layer_past[1].contiguous()
        beam_idx = layer_past[2].contiguous()
        decoded_tokens = layer_past[3].contiguous()
        alibi = (
            alibi.repeat(1, q_length, 1)
            .view(batch_size, self.num_heads, q_length, -1)
            .contiguous()
        )
        (
            context_layer,
            attention_probs,
            key_cache,
            value_cache,
            beam_idx,
        ) = torch.ops.torch_ipex.masked_multihead_self_attention(
            query_layer,
            key_layer,
            value_layer,
            key_cache,
            value_cache,
            beam_idx,
            decoded_tokens,
            1 / self.inv_norm_factor,
            self.text_max_length,
            head_mask,
            attention_mask + alibi if attention_mask is not None else alibi,
        )

        if use_cache is True:
            present = (
                key_cache,
                value_cache,
                beam_idx,
                torch.tensor(layer_past[3] + query_layer.shape[1], dtype=torch.long),
            )
        else:
            present = None

        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

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

        output_tensor = dropout_add(
            output_tensor, residual, self.hidden_dropout, self.training
        )

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class _GLM2Attention(torch.nn.Module):
    """Parallel self-attention layer abstract class.
    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, module, config):
        super(_GLM2Attention, self).__init__()
        self.layer_number = module.layer_number

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

        # Output.
        self.dense = module.dense
        self.factor = (
            self.core_attention.norm_factor
            if self.core_attention.coeff is None
            else self.core_attention.norm_factor / self.core_attention.coeff
        )
        rotary_dim = (
            config.hidden_size // config.num_attention_heads
            if config.kv_channels is None
            else config.kv_channels
        )
        self.pos_emb = create_sinusoidal_positions(config.seq_length, rotary_dim // 2)
        self.text_max_length = (
            config.text_max_length if hasattr(config, "text_max_length") else 2048
        )

    def split_tensor_along_last_dim(
        self,
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
    ):
        """Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                    in memory.

        Returns:
            A list of Tensors
        """
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def forward(
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

        if kv_cache is None:
            kv_cache = (
                torch.randn(0),
                torch.randn(0),
                torch.zeros(2048, 4, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            length = torch.tensor(kv_cache[3][0] + query_layer.shape[1]).contiguous()
            pos_emb = self.pos_emb[:length, ...]
            torch.ops.torch_ipex.rotary_position_embedding(
                key_layer,
                pos_emb,
                kv_cache[3],
                key_layer.size(-2),
                key_layer.size(-1),
                1,
                64,
            )
            torch.ops.torch_ipex.rotary_position_embedding(
                query_layer,
                pos_emb,
                kv_cache[3],
                query_layer.size(-2),
                query_layer.size(-1),
                1,
                64,
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                query_layer.size(0),
                1,
                kv_cache[3] + query_layer.size(1),
                kv_cache[3] + key_layer.size(1),
                dtype=torch.bool,
            )
            attention_mask.tril_()
            attention_mask = ~attention_mask
        query_layer = query_layer.contiguous().type_as(hidden_states)
        key_layer = key_layer.contiguous().type_as(hidden_states)
        value_layer = value_layer.contiguous().type_as(hidden_states)
        key_cache = kv_cache[0].contiguous()
        value_cache = kv_cache[1].contiguous()
        beam_idx = kv_cache[2].contiguous()
        decoded_tokens = kv_cache[3].contiguous()[0]
        (
            context_layer,
            attn_weights,
            key_cache,
            value_cache,
            beam_idx,
        ) = torch.ops.torch_ipex.masked_multihead_self_attention(
            query_layer,
            key_layer,
            value_layer,
            key_cache,
            value_cache,
            beam_idx,
            decoded_tokens,
            self.factor,
            self.text_max_length,
            None,
            attention_mask,
        )
        kv_cache = (
            key_cache,
            value_cache,
            beam_idx,
            torch.tensor(kv_cache[3] + query_layer.shape[1], dtype=torch.long),
        )
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        context_layer = context_layer.reshape(
            context_layer.shape[0], context_layer.shape[1], self.projection_size
        )
        output = self.dense(context_layer)
        return output, kv_cache


class _CodeGenAttention(nn.Module):
    def __init__(self, module, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = module.embed_dim
        self.num_attention_heads = module.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        ).to(torch.get_default_dtype())
        self.qkv_proj = module.qkv_proj

        self.out_proj = module.out_proj
        self.rotary_dim = module.rotary_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)
        self.text_max_length = (
            config.text_max_length if hasattr(config, "text_max_length") else 2048
        )

    def _split_heads(self, x, n_head, dim_head, mp_num):
        reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into n_ctx
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}"
            )
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.causal_mask[
            :, :, key_length - query_length : key_length, :key_length
        ]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / self.scale_attn
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
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

        torch.ops.torch_ipex.rotary_position_embedding(
            key,
            self.embed_positions,
            position_ids,
            self.num_attention_heads,
            self.head_dim,
            1,  # neighbor elements
            64,
        )
        torch.ops.torch_ipex.rotary_position_embedding(
            query,
            self.embed_positions,
            position_ids,
            self.num_attention_heads,
            self.head_dim,
            1,
            64,
        )

        if use_cache:
            if layer_past is None:
                layer_past = (
                    torch.randn(0),
                    torch.randn(0),
                    torch.zeros(2048, key.shape[0], dtype=torch.long),
                    torch.zeros(1, dtype=torch.long),
                )
            key_cache = layer_past[0].contiguous()
            value_cache = layer_past[1].contiguous()
            beam_idx = layer_past[2].contiguous()
            decoded_tokens = layer_past[3].contiguous()[0]
            (
                attn_output,
                attn_weights,
                key_cache,
                value_cache,
                beam_idx,
            ) = torch.ops.torch_ipex.masked_multihead_self_attention(
                query,
                key,
                value,
                key_cache,
                value_cache,
                beam_idx,
                decoded_tokens,
                self.scale_attn,
                self.text_max_length,
                head_mask,
                attention_mask,
            )
            present = (
                key_cache,
                value_cache,
                beam_idx,
                torch.tensor(layer_past[3] + query.shape[1], dtype=torch.long),
            )
        else:
            key = key.permute(0, 2, 1, 3)
            query = query.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)

            if layer_past is not None:
                past_key = layer_past[0]
                past_value = layer_past[1]
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

            if use_cache is True:
                present = (key, value)
            else:
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


class _GPTBigCodeAttention(nn.Module):
    def __init__(self, module, config):
        super().__init__()
        self.mask_value = None

        self.multi_query = module.multi_query
        self.embed_dim = module.embed_dim
        self.num_heads = module.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.kv_heads = 1 if self.multi_query else self.num_heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = module.scale_attn_weights
        self.is_cross_attention = module.is_cross_attention

        self.layer_idx = module.layer_idx
        self.attention_softmax_in_fp32 = module.attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = module.scale_attention_softmax_in_fp32

        if self.is_cross_attention:
            if self.multi_query:
                raise NotImplementedError(
                    "Multi-Query Attention not supported for cross_attention"
                )

            self.c_attn = module.c_attn
            self.q_attn = module.q_attn
        else:
            self.c_attn = module.c_attn

        dtype = self.c_attn.weight.dtype
        softmax_dtype = torch.float32 if self.attention_softmax_in_fp32 else dtype
        unscale = (
            self.layer_idx + 1
            if self.scale_attention_softmax_in_fp32 and dtype != softmax_dtype
            else 1
        )
        scale_factor = 1 / unscale**-1
        if self.scale_attn_weights:
            scale_factor *= self.head_dim**0.5
        self.scale_factor = scale_factor / unscale

        self.c_proj = module.c_proj

        self.attn_dropout = module.attn_dropout
        self.resid_dropout = module.resid_dropout
        self.text_max_length = (
            config.text_max_length if hasattr(config, "text_max_length") else 2048
        )

    def _get_mask_value(self, device, dtype):
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if (
            self.mask_value is None
            or self.mask_value.dtype != dtype
            or self.mask_value.device != device
        ):
            self.mask_value = torch.full(
                [], torch.finfo(dtype).min, dtype=dtype, device=device
            )
        return self.mask_value

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        dtype = query.dtype
        softmax_dtype = torch.float32 if self.attention_softmax_in_fp32 else dtype
        upcast = dtype != softmax_dtype

        unscale = (
            self.layer_idx + 1 if self.scale_attention_softmax_in_fp32 and upcast else 1
        )
        scale_factor = unscale**-1
        if self.scale_attn_weights:
            scale_factor /= self.head_dim**0.5

        # MQA models: (batch_size, query_length, num_heads * head_dim)
        # MHA models: (batch_size, num_heads, query_length, head_dim)
        query_shape = query.shape
        batch_size = query_shape[0]
        key_length = key.size(-1)
        if self.multi_query:
            # (batch_size, query_length, num_heads, head_dim) x (batch_size, head_dim, key_length)
            # -> (batch_size, query_length, num_heads, key_length)
            query_length = query_shape[1]
            attn_shape = (batch_size, query_length, self.num_heads, key_length)
            attn_view = (batch_size, query_length * self.num_heads, key_length)
            # No copy needed for MQA 2, or when layer_past is provided.
            query = query.reshape(
                batch_size, query_length * self.num_heads, self.head_dim
            )
        else:
            # (batch_size, num_heads, query_length, head_dim) x (batch_size, num_heads, head_dim, key_length)
            # -> (batch_size, num_heads, query_length, key_length)
            query_length = query_shape[2]
            attn_shape = (batch_size, self.num_heads, query_length, key_length)
            attn_view = (batch_size * self.num_heads, query_length, key_length)
            # Always copies
            query = query.reshape(
                batch_size * self.num_heads, query_length, self.head_dim
            )
            # No copy when layer_past is provided.
            key = key.reshape(batch_size * self.num_heads, self.head_dim, key_length)

        attn_weights = torch.empty(attn_view, device=query.device, dtype=query.dtype)
        if query.device.type == "cpu":
            # This is needed because of a bug in pytorch https://github.com/pytorch/pytorch/issues/80588.
            # The bug was fixed in https://github.com/pytorch/pytorch/pull/96086,
            # but the fix has not been released as of pytorch version 2.0.0.
            attn_weights = torch.zeros_like(attn_weights)
            beta = 1
        else:
            beta = 0
        attn_weights = torch.baddbmm(
            attn_weights, query, key, beta=beta, alpha=scale_factor
        ).view(attn_shape)

        if upcast:
            # Use a fused kernel to prevent a large overhead from casting and scaling.
            # Sub-optimal when the key length is not a multiple of 8.
            if attention_mask is None:
                attn_weights = upcast_softmax(attn_weights, unscale, softmax_dtype)
            else:
                mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)
                attn_weights = upcast_masked_softmax(
                    attn_weights, attention_mask, mask_value, unscale, softmax_dtype
                )
        else:
            if attention_mask is not None:
                mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)

                # The fused kernel is very slow when the key length is not a multiple of 8, so we skip fusion.
                attn_weights = torch.where(attention_mask, attn_weights, mask_value)

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            if self.multi_query:
                head_mask = head_mask.transpose(1, 2)
            attn_weights = attn_weights * head_mask

        if self.multi_query:
            attn_output = torch.bmm(attn_weights.view(attn_view), value).view(
                query_shape
            )
        else:
            attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn") or not self.is_cross_attention:
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPTBigCodeAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(
                (self.head_dim, self.head_dim), dim=-1
            )
            attention_mask = encoder_attention_mask
        elif self.multi_query:
            query, key, value = self.c_attn(hidden_states).split(
                (self.embed_dim, self.head_dim, self.head_dim), dim=2
            )
        else:
            # Note: We split as (self.num_heads, 3, self.head_dim) instead of (3, self.num_heads, self.head_dim),
            # i.e., the memory layout is not the same as GPT2.
            # This makes the concatenation with past_key_value more efficient.
            query, key, value = (
                self.c_attn(hidden_states)
                .view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim)
                .transpose(1, 2)
                .split((self.head_dim, self.head_dim, self.head_dim), dim=3)
            )

        if layer_past is None:
            layer_past = (
                torch.randn(0),
                torch.randn(0),
                torch.zeros(2048, 4, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )

        batch_size = query.shape[0]
        key_length = key.size(-1)
        query_length = query.shape[1] if self.multi_query else query.shape[2]
        query = query.reshape(
            batch_size, query_length, self.num_heads, self.head_dim
        ).contiguous()
        key = key.reshape(batch_size, query_length, -1, key_length).contiguous()
        value = value.reshape(batch_size, query_length, -1, key_length).contiguous()
        key_cache = layer_past[0].contiguous()
        value_cache = layer_past[1].contiguous()
        beam_idx = layer_past[2].contiguous()
        decoded_tokens = layer_past[3].contiguous()[0]
        (
            attn_output,
            attn_weights,
            key_cache,
            value_cache,
            beam_idx,
        ) = torch.ops.torch_ipex.masked_multihead_self_attention(
            query,
            key,
            value,
            key_cache,
            value_cache,
            beam_idx,
            decoded_tokens,
            self.scale_factor,
            self.text_max_length,
            head_mask,
            attention_mask,
        )
        present = (
            key_cache,
            value_cache,
            beam_idx,
            torch.tensor(layer_past[3] + query.shape[1], dtype=torch.long),
        )

        attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            if self.multi_query:
                # Transpose to return weights in the usual format (batch_size, num_heads, query_length, key_length)
                attn_weights = attn_weights.transpose(1, 2)
            outputs += (attn_weights,)
        return outputs  # a, present, (attentions)


class _BaichuanAttention(torch.nn.Module):
    def __init__(self, module, config):
        super().__init__()
        self.config = config
        self.hidden_size = module.hidden_size
        self.num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.max_position_embeddings = module.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        self.W_pack = module.W_pack
        self.o_proj = module.o_proj
        self.text_max_length = (
            config.text_max_length if hasattr(config, "text_max_length") else 2048
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        proj = self.W_pack(hidden_states)
        proj = proj.split(
            [self.hidden_size, self.hidden_size, self.hidden_size], dim=-1
        )

        query = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).contiguous()
        key = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).contiguous()
        value = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).contiguous()
        if attention_mask is not None:
            if len(attention_mask.size()) == 4:
                attention_mask = attention_mask[:, :, -1:, :]
            else:
                attention_mask = attention_mask[:, -1:, :]
        if past_key_value is None:
            past_key_value = (
                torch.randn(0),
                torch.randn(0),
                torch.zeros(2048, 4, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )
        key_cache = past_key_value[0].contiguous()
        value_cache = past_key_value[1].contiguous()
        beam_idx = past_key_value[2].contiguous()
        decoded_tokens = past_key_value[3].contiguous()[0]
        (
            context_layer,
            attn_weights,
            key_cache,
            value_cache,
            beam_idx,
        ) = torch.ops.torch_ipex.masked_multihead_self_attention(
            query,
            key,
            value,
            key_cache,
            value_cache,
            beam_idx,
            decoded_tokens,
            math.sqrt(self.head_dim),
            self.text_max_length,
            None,
            attention_mask,
        )
        present = (
            key_cache,
            value_cache,
            beam_idx,
            torch.tensor(past_key_value[3] + query.shape[1], dtype=torch.long),
        )
        attn_output = context_layer.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present


class _T5Attention(nn.Module):
    def __init__(self, module, config):
        super().__init__()
        self.is_decoder = module.is_decoder
        self.has_relative_attention_bias = module.has_relative_attention_bias
        self.relative_attention_num_buckets = module.relative_attention_num_buckets
        self.relative_attention_max_distance = module.relative_attention_max_distance
        self.d_model = module.d_model
        self.key_value_proj_dim = module.key_value_proj_dim
        self.n_heads = module.n_heads
        self.dropout = module.dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = module.q
        self.k = module.k
        self.v = module.v
        self.o = module.o

        if self.has_relative_attention_bias:
            self.relative_attention_bias = module.relative_attention_bias
        self.pruned_heads = set()
        self.gradient_checkpointing = False
        self.text_max_length = (
            config.text_max_length if hasattr(config, "text_max_length") else 2048
        )

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/
        mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=True,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                hidden_states = shape(proj_layer(hidden_states))
            else:
                if past_key_value is None:
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    hidden_states = past_key_value.transpose(0, 1)[
                        :, : key_value_states.shape[1], :, :
                    ]
            return hidden_states

        # get query states
        query = shape(self.q(hidden_states))

        # get key/value states
        key = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        ).to(dtype=query.dtype)

        value = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        ).to(dtype=query.dtype)

        real_seq_length = seq_length
        if past_key_value is not None:
            if len(past_key_value) == 2:
                real_seq_length += (
                    past_key_value[0].shape[2] if query_length is None else query_length
                )
            else:
                real_seq_length += (
                    past_key_value[3][0] if query_length is None else query_length
                )

        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape[1]
        )
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=hidden_states.device
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = torch.tensor(position_bias) + torch.tensor(
                    mask
                )  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        if past_key_value is None:
            past_key_value = (
                torch.randn(0),
                torch.randn(0),
                torch.zeros(2048, batch_size, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )
        key_cache = past_key_value[0].contiguous()
        value_cache = past_key_value[1].contiguous()
        beam_idx = past_key_value[2].contiguous()
        if key_value_states is None:
            decoded_tokens = past_key_value[3].contiguous()[0]
        else:
            decoded_tokens = torch.zeros(1, dtype=torch.long).contiguous()[0]
        (
            context_layer,
            attn_weights,
            key_cache,
            value_cache,
            beam_idx,
        ) = torch.ops.torch_ipex.masked_multihead_self_attention(
            query,
            key,
            value,
            key_cache,
            value_cache,
            beam_idx,
            decoded_tokens,
            1,
            self.text_max_length,
            layer_head_mask,
            position_bias_masked,
            False,
        )
        present_key_value_state = (
            (
                key_cache,
                value_cache,
                beam_idx,
                torch.tensor(past_key_value[3] + query.shape[1], dtype=torch.long),
            )
            if (self.is_decoder and use_cache)
            else None
        )
        attn_output = context_layer.transpose(1, 2)
        attn_output = attn_output.reshape(query.shape[0], query.shape[1], -1)
        attn_output = self.o(attn_output)

        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


def _reorder_cache(
    self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
) -> Tuple[Tuple[torch.Tensor]]:
    if len(past_key_values[0]) == 4:  # discrete kv_cache
        for layer_past in past_key_values:
            layer_past[2][layer_past[3] - 1] = beam_idx
        return past_key_values
    elif len(past_key_values[0]) == 8:
        for layer_past in past_key_values:
            layer_past[2][layer_past[3] - 1] = beam_idx
            layer_past[6][layer_past[7] - 1] = beam_idx
        return past_key_values
    else:
        return tuple(layer_past + (beam_idx,) for layer_past in past_key_values)
