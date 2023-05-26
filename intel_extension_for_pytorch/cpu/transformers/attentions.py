import torch
from torch import nn
from typing import Optional, Tuple, Union
import math
from transformers.models.llama.configuration_llama import LlamaConfig


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


def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum(
        "i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq
    ).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


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

        self.k_proj = module.k_proj
        self.v_proj = module.v_proj
        self.q_proj = module.q_proj
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
                beam_idx_tmp = torch.zeros((2048, int(4)), dtype=torch.int).contiguous()
                past_key_value = tuple(
                    [
                        torch.zeros([1, 32, 1, 128]).contiguous(),
                        torch.zeros([1, 32, 1, 128]).contiguous(),
                        beam_idx_tmp,
                        torch.zeros(1, dtype=torch.long).contiguous(),
                    ]
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
        self.num_kv_heads = (
            self.config.num_attention_kv_heads
            if hasattr(self.config, "num_attention_kv_heads")
            else module.num_attention_heads
        )
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.n_rep = self.num_heads // self.num_kv_heads
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = module.q_proj
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.k_proj.weight = module.k_proj.weight
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
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

    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
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
        key = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        )
        position_ids = position_ids.contiguous()
        key = key.contiguous()
        query = query.contiguous()
        torch.ops.torch_ipex.rotary_position_embedding(
            key,
            self.rotary_emb,
            position_ids,
            self.num_kv_heads,
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
                beam_idx_tmp = torch.zeros((2048, int(4)), dtype=torch.int).contiguous()
                past_key_value = tuple(
                    [
                        torch.zeros([1, 32, 1, 128]).contiguous(),
                        torch.zeros([1, 32, 1, 128]).contiguous(),
                        beam_idx_tmp,
                        torch.zeros(1, dtype=torch.long).contiguous(),
                    ]
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
            # repeat k/v heads if n_kv_heads < n_heads
            key = self.repeat_kv(
                key, self.n_rep
            )  # (bs, seqlen, n_local_heads, head_dim)
            value = self.repeat_kv(
                value, self.n_rep
            )  # (bs, seqlen, n_local_heads, head_dim)
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

    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
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

    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
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
                    torch.randn(0),
                    torch.randn(0),
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


def _reorder_cache(
    self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
) -> Tuple[Tuple[torch.Tensor]]:
    if len(past_key_values[0]) == 4:  # discrete kv_cache
        for layer_past in past_key_values:
            layer_past[2][layer_past[3] - 1] = beam_idx
        return past_key_values
    else:
        return tuple(layer_past + (beam_idx,) for layer_past in past_key_values)
