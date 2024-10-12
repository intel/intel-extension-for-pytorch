import torch
from torch import nn
from typing import Optional, Tuple
import math
from torch.nn import functional as F


@torch.library.impl("myops::longrope", "CPU")
def longrope(
    inv_freq,
    max_seq_len_cached,
    max_position_embeddings,
    sin_cos,
    sin_cached,
    cos_cached,
    sin_cos_long,
    sin_cached_long,
    cos_cached_long,
    seq_len,
    rope_type,
):
    if seq_len > max_seq_len_cached:
        if rope_type == 1:  # Phi3ForCausalLM
            return (
                max_position_embeddings,
                sin_cos_long,
                sin_cached_long,
                cos_cached_long,
            )
        elif rope_type == 2:  # Falcon
            t = torch.arange(seq_len, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            sin_cos = torch.cat(
                (freqs.sin().repeat(1, 2), freqs.cos().repeat(1, 2)), dim=-1
            )
            emb = torch.cat((freqs, freqs), dim=-1).float()
            cos_cached = emb.cos()[None, :, :]
            sin_cached = emb.sin()[None, :, :]
            return seq_len, sin_cos, sin_cached, cos_cached
        else:  # Default
            t = torch.arange(seq_len, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            sin_cos = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=1)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos_cached = emb.cos()[None, None, :, :]
            sin_cached = emb.sin()[None, None, :, :]
            return (
                seq_len,
                sin_cos,
                sin_cached[:, :, :seq_len, ...],
                cos_cached[:, :, :seq_len, ...],
            )
    return max_seq_len_cached, sin_cos, sin_cached, cos_cached


torch.library.define(
    "myops::longrope",
    "(Tensor inv_freq, Tensor max_seq_len_cached, Tensor max_position_embeddings, Tensor sin_cos, "
    + " Tensor sin_cached, Tensor cos_cached, Tensor? sin_cos_long, Tensor? sin_cached_long, "
    + "Tensor? cos_cached_long,  Tensor seq_len, Tensor rope_type) -> (Tensor, Tensor, Tensor, Tensor)",
)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, max_position_embeddings, dim, backbone, base=10000, kwargs=None):
        super().__init__()
        self.scaling_factor = 1.0
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len_cached = max_position_embeddings
        if kwargs is not None and "short_factor" in kwargs:
            self.short_factor = kwargs["short_factor"]
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32)
            inv_freq = 1.0 / (
                ext_factors * base ** (torch.arange(0, dim, 2).float() / dim)
            )
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        if kwargs is not None and "long_factor" in kwargs:
            self.long_factor = kwargs["long_factor"]
            ext_factors_long = torch.tensor(self.long_factor, dtype=torch.float32)
            inv_freq_long = 1.0 / (
                ext_factors_long * base ** (torch.arange(0, dim, 2).float() / dim)
            )
        if (
            kwargs is not None
            and "original_max_position_embeddings" in kwargs
            and not ("rope_type" in kwargs and kwargs["rope_type"] == "llama3")
        ):
            self.original_max_position_embeddings = kwargs[
                "original_max_position_embeddings"
            ]
            scale = max_position_embeddings / self.original_max_position_embeddings
            if scale > 1.0:
                if "type" in kwargs and kwargs["type"] == "su":
                    self.scaling_factor = math.sqrt(
                        1
                        + math.log(scale)
                        / math.log(self.original_max_position_embeddings)
                    )
                elif "type" in kwargs and kwargs["type"] == "yarn":
                    self.scaling_factor = 0.1 * math.log(scale) + 1.0
            self.max_seq_len_cached = self.original_max_position_embeddings
        if (
            kwargs is not None
            and "rope_type" in kwargs
            and kwargs["rope_type"] == "llama3"
        ):
            # Values obtained from grid search
            scale_factor = kwargs["factor"]
            low_freq_factor = kwargs["low_freq_factor"]
            high_freq_factor = kwargs["high_freq_factor"]
            old_context_len = kwargs["original_max_position_embeddings"]

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor
            new_freqs = []
            for freq in inv_freq:
                wavelen = 2 * math.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / scale_factor)
                else:
                    assert low_freq_wavelen != high_freq_wavelen
                    smooth = (old_context_len / wavelen - low_freq_factor) / (
                        high_freq_factor - low_freq_factor
                    )
                    new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
            inv_freq = torch.tensor(
                new_freqs, dtype=inv_freq.dtype, device=inv_freq.device
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        if backbone == "Phi3ForCausalLM" and "long_factor" not in kwargs:
            self.max_seq_len_cached = self.max_seq_len_cached + 256
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
            self.sin_cos = (
                torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=1)
                * self.scaling_factor
            )
            self.emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer(
                "cos_cached",
                self.emb.cos()[None, None, :, :] * self.scaling_factor,
                persistent=False,
            )
            self.register_buffer(
                "sin_cached",
                self.emb.sin()[None, None, :, :] * self.scaling_factor,
                persistent=False,
            )
            if hasattr(self, "long_factor"):
                t_long = torch.arange(
                    max_position_embeddings, dtype=self.inv_freq.dtype
                )
                freqs_long = torch.einsum("i,j->ij", t_long, inv_freq_long)
                self.sin_cos_long = (
                    torch.cat((torch.sin(freqs_long), torch.cos(freqs_long)), dim=1)
                    * self.scaling_factor
                )
                self.emb_long = torch.cat((freqs_long, freqs_long), dim=-1)
                self.register_buffer(
                    "cos_cached_long",
                    self.emb_long.cos()[None, None, :, :] * self.scaling_factor,
                    persistent=False,
                )
                self.register_buffer(
                    "sin_cached_long",
                    self.emb_long.sin()[None, None, :, :] * self.scaling_factor,
                    persistent=False,
                )

    def forward(self, seq_len=None):
        rope_type = 0
        if self.model_backbone == "Phi3ForCausalLM" and hasattr(self, "long_factor"):
            rope_type = 1
        elif self.model_backbone in ["FalconForCausalLM", "RWForCausalLM"]:
            rope_type = 2
        if seq_len is not None:
            max_seq_len_cached, self.sin_cos, self.sin_cached, self.cos_cached = (
                torch.ops.myops.longrope(
                    torch.tensor(self.inv_freq).contiguous(),
                    torch.tensor(self.max_seq_len_cached).contiguous(),
                    torch.tensor(self.max_position_embeddings).contiguous(),
                    self.sin_cos.contiguous(),
                    self.sin_cached.contiguous(),
                    self.cos_cached.contiguous(),
                    self.sin_cos_long.contiguous() if rope_type == 1 else None,
                    self.sin_cached_long.contiguous() if rope_type == 1 else None,
                    self.cos_cached_long.contiguous() if rope_type == 1 else None,
                    torch.tensor(seq_len).contiguous(),
                    torch.tensor(rope_type).contiguous(),
                )
            )
            self.max_seq_len_cached = max_seq_len_cached.item()
        return self.sin_cos, self.sin_cached, self.cos_cached


class _IPEXRopeRef(nn.Module):
    def __init__(
        self,
        max_position_embeddings,
        pos_embd_dim,
        base=10000,
        backbone=None,
        kwargs=None,
    ):
        super().__init__()
        self.model_backbone = backbone
        self.embed_positions = RotaryEmbedding(
            max_position_embeddings, pos_embd_dim, backbone, base, kwargs
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
        elif self.model_backbone in [
            "LlamaForCausalLM",
            "MllamaForConditionalGeneration",
            "MistralForCausalLM",
            "MixtralForCausalLM",
            "LlavaLlamaForCausalLM",
            "YuanForCausalLM",
        ]:
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
        elif self.model_backbone in ["StableLmForCausalLM", "PhiForCausalLM"]:
            x = x.transpose(1, 2)
            x_rot = x[..., :rotary_ndims]
            x_pass = x[..., rotary_ndims:]
            x = torch.cat(
                (
                    self.apply_rotary_pos_emb_llama(x_rot, _cos, _sin, position_ids),
                    x_pass,
                ),
                dim=-1,
            )
        elif self.model_backbone == "Phi3ForCausalLM":
            x = x.view(x.shape[0], -1, num_head, head_dim)
            x = x.transpose(1, 2)
            cos = _cos[..., seq_len - x.shape[2] : seq_len, :]
            sin = _sin[..., seq_len - x.shape[2] : seq_len, :]
            x = (x * cos) + (self.rotate_half(x) * sin)
        elif self.model_backbone == "QWenLMHeadModel":
            x = x.view(x.size(0), x.size(1), num_head, head_dim)
            b, sq, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
            x_float = x.float()
            x_rot = x_float[..., :rotary_ndims]
            x_pass = x_float[..., rotary_ndims:]
            sin = (
                _sin.squeeze(1)
                .squeeze(0)[position_ids : position_ids + sq, :]
                .unsqueeze(1)
                .unsqueeze(0)
            )
            cos = (
                _cos.squeeze(1)
                .squeeze(0)[position_ids : position_ids + sq, :]
                .unsqueeze(1)
                .unsqueeze(0)
            )
            x_rot = (x_rot * cos) + (self.rotate_half(x_rot) * sin)
            x = torch.cat((x_rot, x_pass), dim=-1).type_as(x)
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
            self.bias = module.bias if hasattr(module, "bias") else None
            self.scale_attn = module.scale_attn
            self.attn_dropout = module.attn_dropout
        elif self.model_backbone in [
            "LlamaForCausalLM",
            "MllamaForConditionalGeneration",
            "MistralForCausalLM",
            "MixtralForCausalLM",
            "StableLmForCausalLM",
            "LlavaLlamaForCausalLM",
            "YuanForCausalLM",
            "PhiForCausalLM",
            "Phi3ForCausalLM",
        ]:
            self.num_key_value_groups = (
                module.num_key_value_groups
                if hasattr(module, "num_key_value_groups")
                else None
            )
            if hasattr(module, "num_heads"):
                self.num_heads = module.num_heads
            if hasattr(module, "head_dim"):
                self.head_dim = module.head_dim
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
            if hasattr(module, "causal_mask"):
                self.causal_mask = module.causal_mask
            else:
                max_positions = config.max_position_embeddings
                self.causal_mask = torch.tril(
                    torch.ones((max_positions, max_positions), dtype=torch.bool)
                ).view(1, 1, max_positions, max_positions)
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
        elif self.model_backbone == "QWenLMHeadModel":
            self.softmax_in_fp32 = config.softmax_in_fp32
            self.head_dim = module.head_dim
            self.num_heads = module.num_heads
        elif self.model_backbone == "GitForCausalLM":
            if hasattr(module, "num_heads"):
                self.num_heads = module.num_heads
            if hasattr(module, "head_dim"):
                self.head_dim = module.head_dim

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
        cutoff: Optional[torch.Tensor] = None,
        vision: Optional[torch.Tensor] = False,
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
                    "QWenLMHeadModel",
                    "GitForCausalLM",
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
                    "QWenLMHeadModel",
                    "GitForCausalLM",
                ]
                else query
            )
            if self.model_backbone == "QWenLMHeadModel":
                value = value.view(
                    value.size(0), value.size(1), self.num_heads, self.head_dim
                )
            value = value.permute(0, 2, 1, 3)

        if self.model_backbone == "GitForCausalLM":
            if not (vision is not None and vision):
                if layer_past is not None:
                    key = torch.cat(
                        [key[:, :, :cutoff, :], layer_past[0], key[:, :, -1:, :]], dim=2
                    )
                    value = torch.cat(
                        [value[:, :, :cutoff, :], layer_past[1], value[:, :, -1:, :]],
                        dim=2,
                    )
            present = (
                key[:, :, cutoff:, :],
                value[:, :, cutoff:, :],
            )
        elif (
            self.model_backbone == "LlavaLlamaForCausalLM"
            and vision is not None
            and vision
        ):
            present = None
        else:
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
        elif self.model_backbone in [
            "LlamaForCausalLM",
            "MllamaForConditionalGeneration",
            "MistralForCausalLM",
            "MixtralForCausalLM",
            "StableLmForCausalLM",
            "YuanForCausalLM",
            "PhiForCausalLM",
            "Phi3ForCausalLM",
        ]:
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
        elif self.model_backbone == "QWenLMHeadModel":
            key_size = key.size(2)
            if query.size(2) == key.size(2):
                causal_mask = torch.tril(
                    torch.ones(
                        (key_size, key_size), dtype=torch.bool, device=query.device
                    )
                ).view(1, 1, key_size, key_size)
            else:
                causal_mask = None
            attn_weights = torch.matmul(query, key.transpose(-1, -2)) / scale_attn
            mask_value = torch.finfo(attn_weights.dtype).min
            if causal_mask is not None:
                attn_weights = torch.where(
                    causal_mask, attn_weights.to(attn_weights.dtype), mask_value
                )

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            if self.softmax_in_fp32:
                attn_weights = nn.functional.softmax(attn_weights.float(), dim=-1)
            else:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            attn_weights = attn_weights.type(query.dtype)

            if head_mask is not None:
                attn_weights = attn_weights * head_mask

            attn_output = torch.matmul(attn_weights, value)
        elif self.model_backbone == "GitForCausalLM":
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / scale_attn
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            # Normalize the attention scores to probabilities.
            attn_weights = nn.functional.softmax(attention_scores, dim=-1)
            # Mask heads if we want to
            if head_mask is not None:
                attn_weights = attn_weights * head_mask
            attn_output = torch.matmul(attn_weights, value)
        elif self.model_backbone == "LlavaLlamaForCausalLM":
            if vision is not None and vision:
                bsz = query.shape[0]
                proj_shape = (bsz * self.num_heads, -1, self.head_dim)
                query_states = query.transpose(1, 2).contiguous().view(*proj_shape)
                key_states = key.transpose(1, 2).contiguous().view(*proj_shape)
                value_states = value.view(*proj_shape)

                src_len = key_states.size(1)
                attn_weights = (
                    torch.bmm(query_states, key_states.transpose(1, 2)) / scale_attn
                )

                if attention_mask is not None:
                    attn_weights = (
                        attn_weights.view(bsz, self.num_heads, -1, src_len)
                        + attention_mask
                    )
                    attn_weights = attn_weights.view(bsz * self.num_heads, -1, src_len)

                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
                attn_output = torch.bmm(attn_weights, value_states)
                attn_weights = attn_weights.view(bsz, self.num_heads, -1, src_len)
            else:
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
        else:
            AssertionError(False, "Do not support the optimization of your model yet")
        return attn_output, attn_weights, present


class _IPEXRMSNormRef(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.weight = module.weight
        self.variance_epsilon = module.variance_epsilon

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class _IPEXPagedAttentionRef:
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        if key.dtype in [torch.bfloat16, torch.float16]:
            x = 16 // torch.tensor([], dtype=key.dtype).element_size()
        else:
            x = 32 // torch.tensor([], dtype=key.dtype).element_size()

        num_key_value_heads = key.shape[-2]
        head_size = key.shape[-1]
        reshaped_key = key.reshape(-1, num_key_value_heads, head_size // x, x)
        num_tokens = value.shape[0]
        block_size = value_cache.shape[3]
        for i in range(num_tokens):
            block_idx = torch.div(slot_mapping[i], block_size, rounding_mode="floor")
            block_offset = slot_mapping[i] % block_size
            key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
            value_cache[block_idx, :, :, block_offset] = value[i]

    @classmethod
    def single_query_cached_kv_attention(
        cls,
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
    ) -> None:
        num_heads = value_cache.shape[1]
        head_size = value_cache.shape[2]
        block_size = value_cache.shape[3]

        num_input_tokens = query.shape[0]
        for i in range(num_input_tokens):
            q = query[i].unsqueeze(0)
            block_table = block_tables[i]
            context_len = int(context_lens[i])
            keys = []
            values = []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = key_cache[block_number, :, :, block_offset, :]
                k = k.reshape(num_heads, head_size)
                keys.append(k)

                v = value_cache[block_number, :, :, block_offset]
                values.append(v)
            keys = torch.stack(keys, dim=0)
            values = torch.stack(values, dim=0)

            scale = 1.0 / (head_size**0.5)
            out = torch.nn.functional.scaled_dot_product_attention(
                q.view(1, -1, num_heads, head_size).transpose(1, 2),
                keys.view(1, -1, num_heads, head_size).transpose(1, 2),
                values.view(1, -1, num_heads, head_size).transpose(1, 2),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            out = out.view(num_heads, head_size)
            output[i].copy_(out, non_blocking=True)
