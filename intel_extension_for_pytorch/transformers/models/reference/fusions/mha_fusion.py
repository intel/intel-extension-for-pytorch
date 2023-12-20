import torch
from torch import nn
from typing import Optional, Tuple
import math
from torch.nn import functional as F


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, max_position_embeddings, dim, backbone, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            dtype=self.inv_freq.dtype,
        )
        self.model_backbone = str(backbone)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        if (
            self.model_backbone == "FalconForCausalLM"
            or self.model_backbone == "RWForCausalLM"
        ):
            self.sin_cos = torch.cat(
                (freqs.sin().repeat(1, 2), freqs.cos().repeat(1, 2)), dim=-1
            )
            self.emb = torch.cat((freqs, freqs), dim=-1).float()
            self.cos_cached = self.emb.cos()[None, :, :]
            self.sin_cached = self.emb.sin()[None, :, :]
        else:
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
            if (
                self.model_backbone == "FalconForCausalLM"
                or self.model_backbone == "RWForCausalLM"
            ):
                self.sin_cos = torch.cat(
                    (freqs.sin().repeat(1, 2), freqs.cos().repeat(1, 2)), dim=-1
                )
                self.emb = torch.cat((freqs, freqs), dim=-1).float()
                self.cos_cached = self.emb.cos()[None, :, :]
                self.sin_cached = self.emb.sin()[None, :, :]
            else:
                self.sin_cos = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=1)
                self.emb = torch.cat((freqs, freqs), dim=-1)
                self.cos_cached = self.emb.cos()[None, None, :, :]
                self.sin_cached = self.emb.sin()[None, None, :, :]
                self.cos_cached[:, :, :seq_len, ...]
                self.sin_cached[:, :, :seq_len, ...]
        return self.sin_cos, self.sin_cached, self.cos_cached


class _IPEXRopeRef(nn.Module):
    def __init__(
        self,
        max_position_embeddings,
        pos_embd_dim,
        base=10000,
        backbone=None,
    ):
        super().__init__()
        self.model_backbone = backbone
        self.embed_positions = RotaryEmbedding(
            max_position_embeddings, pos_embd_dim, backbone, base
        )

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

    def apply_rotary_pos_emb_baichuan(self, x, cos, sin, position_ids):
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        x_embed = (x.float() * cos) + (self.rotate_half(x.float()) * sin)
        return x_embed.to(x.dtype)

    def apply_ref_rope(
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
        if self.model_backbone == "GPTJForCausalLM":
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
        elif self.model_backbone in ["LlamaForCausalLM", "MistralForCausalLM"]:
            x = x.transpose(1, 2)
            x = self.apply_rotary_pos_emb_llama(x, _cos, _sin, position_ids)
        elif self.model_backbone == "BaichuanForCausalLM":
            x = self.apply_rotary_pos_emb_baichuan(x, _cos, _sin, position_ids)
        elif self.model_backbone == "GPTNeoXForCausalLM":
            x = x.transpose(1, 2)
            x_rot = x[..., :rotary_ndims]
            x_pass = x[..., rotary_ndims:]
            x = torch.cat(
                (
                    self.apply_rotary_pos_emb_gptneox(x_rot, _cos, _sin, position_ids),
                    x_pass,
                ),
                dim=-1,
            )
        elif (
            self.model_backbone == "FalconForCausalLM"
            or self.model_backbone == "RWForCausalLM"
        ):
            batch_size, x_length, _, _ = x.shape
            x = x.transpose(1, 2).reshape(batch_size * num_head, x_length, head_dim)
            _cos = _cos.type(x.dtype)[:, 0:seq_len]
            _sin = _sin.type(x.dtype)[:, 0:seq_len]
            x = (x * _cos) + (self.rotate_half(x) * _sin)
        elif self.model_backbone == "CodeGenForCausalLM":
            sincos = _sin_cos[position_ids]
            sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
            if rotary_ndims is not None:
                x_rot = x[:, :, :, :rotary_ndims]
                x_pass = x[:, :, :, rotary_ndims:]

                x_rot = self.apply_rotary_pos_emb_gptj(x_rot, sin, cos)
                x = torch.cat([x_rot, x_pass], dim=-1)
            else:
                x = self.apply_rotary_pos_emb_gptj(x, sin, cos)
        elif self.model_backbone == "ChatGLMModel":
            b, sq, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
            x, x_pass = x[..., :rotary_ndims], x[..., rotary_ndims:]
            sincos = _sin_cos[None, None, position_ids : position_ids + sq, :]
            sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
            x = x.reshape(b, -1, np, rotary_ndims // 2, 2)
            x0 = x[..., 0].transpose(1, 2)
            x1 = x[..., 1].transpose(1, 2)
            x_rot = (
                torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
                .flatten(3)
                .transpose(1, 2)
            )
            x = torch.cat([x_rot, x_pass], dim=-1)
        else:
            AssertionError(False, "Do not support the optimization of your model yet")
        return x

    def forward(
        self,
        concat_x: torch.Tensor,
        position_ids: torch.Tensor,
        num_head: int,
        head_dim: int,
        offset: int,
        rotary_ndims: int,
        seq_len: Optional[int] = None,
        num_concats: Optional[int] = None,
    ):
        if num_concats is None:
            return self.apply_ref_rope(
                concat_x,
                position_ids,
                num_head,
                head_dim,
                offset,
                rotary_ndims,
                seq_len,
            )
        else:
            hidden_size = concat_x.shape[-1] // num_concats
            query = concat_x[..., :hidden_size]
            key = concat_x[..., hidden_size : 2 * hidden_size]
            value = concat_x[..., 2 * hidden_size :]
            query = self.apply_ref_rope(
                query,
                position_ids,
                num_head,
                head_dim,
                offset,
                rotary_ndims,
                seq_len,
            )
            key = self.apply_ref_rope(
                key, position_ids, num_head, head_dim, offset, rotary_ndims, seq_len
            )
            return query, key, value


class _IPEXScaleDotProductRef(nn.Module):
    def __init__(self, module, config):
        super().__init__()
        self.model_backbone = config.architectures[0]
        if self.model_backbone == "GPTJForCausalLM":
            self.bias = module.bias
            self.scale_attn = module.scale_attn
            self.attn_dropout = module.attn_dropout
        elif self.model_backbone in ["LlamaForCausalLM", "MistralForCausalLM"]:
            self.num_key_value_groups = (
                module.num_key_value_groups
                if hasattr(module, "num_key_value_groups")
                else None
            )
        elif self.model_backbone == "OPTForCausalLM":
            self.num_heads = module.num_heads
            self.head_dim = module.head_dim
        elif self.model_backbone == "GPTNeoXForCausalLM":
            self.bias = module.bias
            self.norm_factor = module.norm_factor
            self.attention_dropout = (
                module.attention_dropout
                if hasattr(module, "attention_dropout")
                else None
            )
        elif (
            self.model_backbone == "FalconForCausalLM"
            or self.model_backbone == "RWForCausalLM"
        ):
            self.num_heads = module.num_heads
            self.head_dim = module.head_dim
            self.new_decoder_architecture = (
                module.new_decoder_architecture
                if hasattr(module, "new_decoder_architecture")
                else None
            )
        elif self.model_backbone == "BloomForCausalLM":
            self.head_dim = module.head_dim
            self.num_heads = module.num_heads
            self.inv_norm_factor = module.inv_norm_factor
            self.beta = module.beta
            self.attention_dropout = module.attention_dropout
        elif self.model_backbone == "CodeGenForCausalLM":
            self.num_heads = module.num_attention_heads
            self.head_dim = module.head_dim
            self.scale_attn = module.scale_attn
            self.attn_dropout = module.attn_dropout
            self.causal_mask = module.causal_mask
        elif self.model_backbone == "BaichuanForCausalLM":
            self.head_dim = module.head_dim
            self.num_heads = module.num_heads
            self.hidden_size = module.hidden_size
            if hasattr(self, "rotary_emb"):
                self.rotary_emb = module.rotary_emb
        elif self.model_backbone == "ChatGLMModel":
            self.multi_query_attention = module.multi_query_attention
            self.num_attention_heads_per_partition = (
                module.num_attention_heads_per_partition
            )
            self.num_multi_query_groups_per_partition = (
                module.num_multi_query_groups_per_partition
            )
            self.hidden_size_per_attention_head = module.hidden_size_per_attention_head
        elif self.model_backbone == "GPTBigCodeForCausalLM":
            self.scale_attention_softmax_in_fp32 = (
                module.scale_attention_softmax_in_fp32
            )
            self.attention_softmax_in_fp32 = module.attention_softmax_in_fp32
            self.layer_idx = module.layer_idx
            self.scale_attn_weights = module.scale_attn_weights
            self.embed_dim = module.embed_dim
            self.num_heads = module.num_heads
            self.head_dim = self.embed_dim // self.num_heads
            self.multi_query = module.multi_query
            self.mask_value = None
            self.attn_dropout = module.attn_dropout
        elif self.model_backbone == "T5ForConditionalGeneration":
            self.dropout = module.dropout
        elif self.model_backbone == "MptForCausalLM":
            self.hidden_size = module.hidden_size
            self.n_heads = module.n_heads
            self.head_dim = module.head_dim
            self.softmax_scale = module.softmax_scale
            self.attn_dropout_p = module.attn_dropout_p

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
        alibi: Optional[torch.Tensor] = None,
        add_casual_mask: Optional[bool] = True,
        seq_info: Optional[torch.Tensor] = None,
    ):
        if (
            self.model_backbone == "FalconForCausalLM"
            or self.model_backbone == "RWForCausalLM"
        ):
            num_kv_heads = (
                self.num_heads if self.new_decoder_architecture else self.num_kv_heads
            )
            BS_head_num, query_length, _ = query.shape
            batch_size = BS_head_num // self.num_heads
            value = value.transpose(1, 2).reshape(
                batch_size * num_kv_heads, query_length, self.head_dim
            )
        else:
            key = (
                key.permute(0, 2, 1, 3)
                if self.model_backbone
                in [
                    "GPTJForCausalLM",
                    "OPTForCausalLM",
                    "BloomForCausalLM",
                    "CodeGenForCausalLM",
                    "BaichuanForCausalLM",
                    "ChatGLMModel",
                    "GPTBigCodeForCausalLM",
                    "T5ForConditionalGeneration",
                    "MptForCausalLM",
                ]
                else key
            )
            query = (
                query.permute(0, 2, 1, 3)
                if self.model_backbone
                in [
                    "GPTJForCausalLM",
                    "OPTForCausalLM",
                    "BloomForCausalLM",
                    "CodeGenForCausalLM",
                    "BaichuanForCausalLM",
                    "ChatGLMModel",
                    "T5ForConditionalGeneration",
                    "MptForCausalLM",
                ]
                else query
            )
            value = value.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value)

        if self.model_backbone in ["GPTJForCausalLM", "CodeGenForCausalLM"]:
            attn_output, attn_weights = self._attn(
                query, key, value, attention_mask, head_mask
            )
        elif self.model_backbone in ["LlamaForCausalLM", "MistralForCausalLM"]:
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

        elif self.model_backbone == "GPTNeoXForCausalLM":
            # Compute attention
            attn_output, attn_weights = self._attn(
                query, key, value, attention_mask, head_mask
            )
        elif self.model_backbone == "OPTForCausalLM":
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
        elif (
            self.model_backbone == "FalconForCausalLM"
            or self.model_backbone == "RWForCausalLM"
        ):
            _, kv_length, _ = key.shape
            attention_mask_float = attention_mask
            query_layer_ = query.reshape(batch_size, self.num_heads, -1, self.head_dim)
            key_layer_ = key.reshape(batch_size, num_kv_heads, -1, self.head_dim)
            value_layer_ = value.reshape(batch_size, num_kv_heads, -1, self.head_dim)

            if alibi is None:
                attention_scores = query_layer_ @ key_layer_.transpose(-1, -2)
                attention_scores /= math.sqrt(self.head_dim)
                attention_scores = torch.nn.functional.softmax(
                    attention_scores + attention_mask_float,
                    dim=-1,
                    dtype=torch.float,
                )
                attn_output = attention_scores @ value_layer_

                attn_output = attn_output.view(
                    batch_size, self.num_heads, query_length, self.head_dim
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
                attention_probs = torch.nn.functional.softmax(
                    attention_mask_float,
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

            attn_weights = attention_scores if alibi is None else attention_probs
        elif self.model_backbone == "BloomForCausalLM":
            batch_size, _, q_length, _ = query.shape
            query = query.reshape(batch_size * self.num_heads, q_length, self.head_dim)
            key = key.reshape(batch_size * self.num_heads, -1, self.head_dim)
            value = value.reshape(batch_size * self.num_heads, -1, self.head_dim)
            matmul_result = alibi.baddbmm(
                batch1=query,
                batch2=key.transpose(-1, -2),
                beta=self.beta,
                alpha=self.inv_norm_factor,
            )
            attention_scores = matmul_result.view(
                batch_size, self.num_heads, q_length, -1
            )
            input_dtype = attention_scores.dtype
            if input_dtype == torch.float16:
                attention_scores = attention_scores.to(torch.float)
            new_alibi = (
                alibi.repeat(1, q_length, 1)
                .view(batch_size, self.num_heads, q_length, -1)
                .contiguous()
            )
            attn_weights = torch.masked_fill(
                attention_scores,
                (attention_mask - new_alibi).to(torch.bool),
                torch.finfo(attention_scores.dtype).min,
            )
            attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                input_dtype
            )
            attention_probs = self.attention_dropout(attention_probs)
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            attention_probs_reshaped = attention_probs.view(
                batch_size * self.num_heads, q_length, -1
            )
            attn_output = torch.bmm(attention_probs_reshaped, value)
            attn_output = attn_output.view(
                batch_size, self.num_heads, q_length, self.head_dim
            )
        elif self.model_backbone == "BaichuanForCausalLM":
            attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value)
        elif self.model_backbone == "ChatGLMModel":
            key = key.permute(2, 0, 1, 3)
            value = value.permute(2, 0, 1, 3)
            query = query.permute(2, 0, 1, 3)
            if self.multi_query_attention:
                key = key.unsqueeze(-2)
                key = key.expand(
                    -1,
                    -1,
                    -1,
                    self.num_attention_heads_per_partition
                    // self.num_multi_query_groups_per_partition,
                    -1,
                )
                key = key.contiguous().view(
                    key.size()[:2]
                    + (
                        self.num_attention_heads_per_partition,
                        self.hidden_size_per_attention_head,
                    )
                )
                value = value.unsqueeze(-2)
                value = value.expand(
                    -1,
                    -1,
                    -1,
                    self.num_attention_heads_per_partition
                    // self.num_multi_query_groups_per_partition,
                    -1,
                )
                value = value.contiguous().view(
                    value.size()[:2]
                    + (
                        self.num_attention_heads_per_partition,
                        self.hidden_size_per_attention_head,
                    )
                )
            attention_mask = None
            query, key, value = [k.permute(1, 2, 0, 3) for k in [query, key, value]]
            if query.shape[2] == key.shape[2]:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, is_causal=True
                )
            else:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attention_mask
                )
            attn_weights = None
        elif self.model_backbone == "GPTBigCodeForCausalLM":
            query = query.reshape(*query.shape[:2], -1)
            key = key.squeeze()
            value = value.squeeze()
            attention_mask = attention_mask.transpose(1, 2)
            attn_output, attn_weights = self._attn(
                query, key.transpose(-1, -2), value, attention_mask, head_mask
            )
            if self.multi_query:
                attn_output = attn_output.transpose(1, 2)
        elif self.model_backbone == "T5ForConditionalGeneration":
            # compute scores
            scores = torch.matmul(
                query, key.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query, key), compatible with onnx op>9

            scores += attention_mask
            attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
                scores
            )  # (batch_size, n_heads, seq_length, key_length)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=self.training
            )  # (batch_size, n_heads, seq_length, key_length)

            # Mask heads if we want to
            if head_mask is not None:
                attn_weights = attn_weights * head_mask
            attn_output = torch.matmul(attn_weights, value)
        elif self.model_backbone == "MptForCausalLM":
            attention_scores = (
                torch.matmul(query, key.transpose(-1, -2)) * self.softmax_scale
            )
            if alibi is not None:
                attention_scores = attention_scores + alibi
                attention_mask = (attention_mask - alibi).to(torch.bool)
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(
                    attention_mask, torch.finfo(query.dtype).min
                )
            attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).to(
                value.dtype
            )
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.attn_dropout_p, training=self.training
            )

            attn_output = torch.matmul(attn_weights, value)
        else:
            AssertionError(False, "Do not support the optimization of your model yet")
        return attn_output, attn_weights, present
