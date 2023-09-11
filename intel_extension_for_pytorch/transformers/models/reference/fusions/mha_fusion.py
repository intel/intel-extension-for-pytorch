import torch
from torch import nn
from typing import Optional, Tuple
import re


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, max_position_embeddings, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        self.sin_cos = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=1)
        self.emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", self.emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", self.emb.sin()[None, None, :, :], persistent=False
        )

    def forward(self, seq_len=None):
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            self.sin_cos = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=1)
            self.emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = self.emb.cos()[None, None, :, :]
            self.sin_cached = self.emb.sin()[None, None, :, :]
            self.cos_cached[:, :, :seq_len, ...]
            self.sin_cached[:, :, :seq_len, ...]
        return self.sin_cos, self.cos_cached, self.sin_cached


class _IPEXRopeRef(nn.Module):
    def __init__(
        self,
        max_position_embeddings,
        pos_embd_dim,
        base=10000,
        backbone=None,
    ):
        super().__init__()
        self.embed_positions = RotaryEmbedding(
            max_position_embeddings, pos_embd_dim, base
        )
        self.model_backbone = backbone

    def rotate_every_two(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

    def apply_rotary_pos_emb_gptj(
        self, tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
    ) -> torch.Tensor:
        sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        return (tensor * cos) + (self.rotate_every_two(tensor) * sin)

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb_llama(self, x, cos, sin, position_ids):
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        x_embed = (x * cos) + (self.rotate_half(x) * sin)
        return x_embed

    def apply_rotary_pos_emb_gptneox(self, x, cos, sin, position_ids):
        gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
        gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
        cos = torch.gather(
            cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices
        )
        sin = torch.gather(
            sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices
        )
        x_embed = (x * cos) + (self.rotate_half(x) * sin)
        return x_embed

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        num_head: int,
        head_dim: int,
        offset: int,
        rotary_ndims: int,
        seq_len: Optional[int] = None,
    ):
        _sin_cos, _sin, _cos = self.embed_positions(seq_len)
        if re.search("GPTJ", self.model_backbone, re.IGNORECASE):
            embed_positions = _sin_cos.repeat(position_ids.shape[0], 1, 1)
            repeated_position_ids = position_ids.unsqueeze(-1).repeat(
                1, 1, embed_positions.shape[-1]
            )
            sincos = torch.gather(embed_positions, 1, repeated_position_ids)
            sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
            if rotary_ndims is not None:
                x_rot = x[:, :, :, :rotary_ndims]
                x_pass = x[:, :, :, rotary_ndims:]
                x_rot = self.apply_rotary_pos_emb_gptj(x_rot, sin, cos)
                x = torch.cat([x_rot, x_pass], dim=-1)
            else:
                x = self.apply_rotary_pos_emb_gptj(x, sin, cos)
        elif re.search("llama", self.model_backbone, re.IGNORECASE):
            x = x.transpose(1, 2)
            x = self.apply_rotary_pos_emb_llama(x, _sin, _cos, position_ids)
        elif re.search("gptneox", self.model_backbone, re.IGNORECASE):
            x = x.transpose(1, 2)
            x_rot = x[..., :rotary_ndims]
            x_pass = x[..., rotary_ndims:]
            x = torch.cat(
                (
                    self.apply_rotary_pos_emb_gptneox(x_rot, _sin, _cos, position_ids),
                    x_pass,
                ),
                dim=-1,
            )

        else:
            AssertionError(False, "Do not support the optimization of your model yet")
        return x


class _IPEXScaleDotProductRef(nn.Module):
    def __init__(self, module, config):
        super().__init__()
        self.model_backbone = config.architectures[0]
        if re.search("GPTJ", self.model_backbone, re.IGNORECASE):
            self.bias = module.bias
            self.scale_attn = module.scale_attn
            self.attn_dropout = module.attn_dropout
        elif re.search("llama", self.model_backbone, re.IGNORECASE):
            self.num_key_value_groups = (
                module.num_key_value_groups
                if hasattr(module, "num_key_value_groups")
                else None
            )
        elif re.search("OPT", self.model_backbone, re.IGNORECASE):
            self.num_heads = module.num_heads
            self.head_dim = module.head_dim
        elif re.search("gptneox", self.model_backbone, re.IGNORECASE):
            self.bias = module.bias
            self.norm_factor = module.norm_factor
            self.attention_dropout = (
                module.attention_dropout
                if hasattr(module, "attention_dropout")
                else None
            )

        for k, v in module.__class__.__dict__.items():
            if k.startswith("__") or k.startswith("forward"):
                continue
            setattr(self.__class__, k, getattr(module.__class__, k))

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_attn: float,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[Tuple[torch.Tensor]] = None,
    ):
        key = (
            key.permute(0, 2, 1, 3)
            if re.search("GPTJ", self.model_backbone, re.IGNORECASE)
            or re.search("OPT", self.model_backbone, re.IGNORECASE)
            else key
        )
        query = (
            query.permute(0, 2, 1, 3)
            if re.search("GPTJ", self.model_backbone, re.IGNORECASE)
            or re.search("OPT", self.model_backbone, re.IGNORECASE)
            else query
        )
        value = value.permute(0, 2, 1, 3)
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value)

        if re.search("GPTJ", self.model_backbone, re.IGNORECASE):
            attn_output, attn_weights = self._attn(
                query, key, value, attention_mask, head_mask
            )
        elif re.search("llama", self.model_backbone, re.IGNORECASE):
            # repeat k/v heads if n_kv_heads < n_heads
            key = self._repeat_kv(key, self.num_key_value_groups)
            value = self._repeat_kv(value, self.num_key_value_groups)

            attn_weights = torch.matmul(query, key.transpose(2, 3)) / scale_attn

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query.dtype)
            attn_output = torch.matmul(attn_weights, value)

        elif re.search("gptneox", self.model_backbone, re.IGNORECASE):
            # Compute attention
            attn_output, attn_weights = self._attn(
                query, key, value, attention_mask, head_mask
            )
        elif re.search("OPT", self.model_backbone, re.IGNORECASE):
            bsz, _, tgt_len, _ = query.size()
            proj_shape = (bsz * self.num_heads, -1, self.head_dim)
            query_states = query.view(*proj_shape) / scale_attn
            key_states = key.view(*proj_shape)
            value_states = value.view(*proj_shape)

            src_len = key_states.size(1)
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

            if attention_mask is not None:
                attn_weights = (
                    attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                    + attention_mask
                )
                attn_weights = torch.max(
                    attn_weights,
                    torch.tensor(
                        torch.finfo(attn_weights.dtype).min, device=attn_weights.device
                    ),
                )
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            if attn_weights.dtype == torch.float16:
                attn_weights = nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(torch.float16)
            else:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            if head_mask is not None:
                attn_weights = head_mask.view(1, -1, 1, 1) * attn_weights.view(
                    bsz, self.num_heads, tgt_len, src_len
                )
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            attn_output = torch.bmm(attn_weights, value_states)
        else:
            AssertionError(False, "Do not support the optimization of your model yet")
        return attn_output, attn_weights, present
