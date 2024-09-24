import torch
import torch.nn as nn
from .._transformer_configuration import IPEXTransformerConfig
from .CacheUtils import CacheFormat


class PositionalEmbedding(nn.Module):
    def __init__(self, config: IPEXTransformerConfig, dtype):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.dynamic_cache_stride = config.dynamic_cache_stride

    def forward(self, query, key, position_ids, layer_id, beam_size, kv_seq_len):
        return query, key


class GPTJRotaryEmbeddingRef(PositionalEmbedding):
    def __init__(self, config: IPEXTransformerConfig, dtype):
        super().__init__(config=config, dtype=dtype)
        self.rotary_dim = config.rotary_dim
        self.base = config.positional_embedding_base
        self.device = config.device
        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.embed_positions = self.create_sinusoidal_positions(
            config.max_positions, pos_embd_dim
        )

    def create_sinusoidal_positions(self, num_pos: int, dim: int) -> torch.Tensor:
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=self.dtype) / dim))
        sinusoid_inp = torch.einsum(
            "i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq
        ).float()
        res = torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)
        return res

    def _get_embed_positions(self, position_ids):
        embed_positions = self.embed_positions
        if embed_positions.device != position_ids.device:
            embed_positions = embed_positions.to(position_ids.device)
            self.embed_positions = embed_positions
        return embed_positions.repeat(position_ids.shape[0], 1, 1)

    def rotate_every_two(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

    def apply_rotary_pos_emb(
        self, tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
    ) -> torch.Tensor:
        sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        return (tensor * cos) + (self.rotate_every_two(tensor) * sin)

    def forward(self, query, key, position_ids, layer_id, beam_size, kv_seq_len):
        # position_ids [bs*beam, seq_len]
        first_token = position_ids.shape[-1] > 1
        # beam search first_token
        # query, key shape [bs*beam, seq, hidden_size], layout [bs*beam, seq, hidden_size]
        # greedy search/ beam search 2nd token
        # query, key shape [seq, bs*beam, hidden_size], layout [seq, bs*beam, hidden_size]
        if beam_size == 1 or not first_token:
            query = query.transpose(0, 1).contiguous()
            key = key.transpose(0, 1).contiguous()
            position_ids = position_ids.transpose(0, 1).contiguous()
        seq_len = key.size(1)
        if seq_len >= self.dynamic_cache_stride:
            new_cache_length = seq_len + self.dynamic_cache_stride
            self.embed_positions = self.create_sinusoidal_positions(
                new_cache_length, self.rotary_dim
            )
            self.dynamic_cache_stride = new_cache_length
        embed_positions = self._get_embed_positions(position_ids)
        repeated_position_ids = position_ids.unsqueeze(-1).repeat(
            1, 1, embed_positions.shape[-1]
        )
        sincos = torch.gather(embed_positions, 1, repeated_position_ids)
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]
            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]
            k_rot = self.apply_rotary_pos_emb(k_rot, sin, cos)
            q_rot = self.apply_rotary_pos_emb(q_rot, sin, cos)
            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            key = self.apply_rotary_pos_emb(key, sin, cos)
            query = self.apply_rotary_pos_emb(query, sin, cos)

        if beam_size == 1 or not first_token:
            query = query.transpose(0, 1).contiguous()
            key = key.transpose(0, 1).contiguous()
        return query, key


class GPTJRotaryEmbedding(PositionalEmbedding):
    cos_cached = None
    sin_cached = None
    max_position_embedding = 2048

    def __init__(self, config: IPEXTransformerConfig, dtype):
        super().__init__(config=config, dtype=dtype)
        self.rotary_dim = config.rotary_dim
        self.max_position_embedding = config.max_positions
        self.base = config.positional_embedding_base
        self.device = config.device
        if (
            GPTJRotaryEmbedding.cos_cached is None
            or GPTJRotaryEmbedding.sin_cached is None
        ):
            self.create_sin_cos_cache(GPTJRotaryEmbedding.max_position_embedding)

    def create_sin_cos_cache(self, pos):
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotary_dim, 2).float().to(self.device)
                / self.rotary_dim
            )
        )
        t = torch.arange(pos, dtype=torch.float, device=self.device)
        sinusoid_inp = torch.einsum("i , j -> i j", t, inv_freq).float()
        embed_positions = torch.cat(
            (torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1
        )

        sin, cos = torch.split(embed_positions, embed_positions.shape[-1] // 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, 1).to(self.dtype).to(self.device)
        cos = torch.repeat_interleave(cos, 2, 1).to(self.dtype).to(self.device)
        GPTJRotaryEmbedding.sin_cached = sin
        GPTJRotaryEmbedding.cos_cached = cos

    def rotate_every_two(self, x: torch.Tensor) -> torch.Tensor:
        # the original rotary_every_two funtion used in the model
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

    def apply_rotary_pos_emb(self, query, key, sin, cos):
        torch.ops.torch_ipex.apply_rotary_embedding_two_qk(
            query, key, sin, cos, query, key
        )

    def get_sin_cos(self, position_ids, layer_id, beam_size, kv_seq_len):
        # position_ids [bs*beam, seq_len]
        first_token = position_ids.shape[-1] > 1
        if GPTJRotaryEmbedding.max_position_embedding < kv_seq_len:
            new_cache_length = kv_seq_len + self.dynamic_cache_stride
            self.create_sin_cos_cache(new_cache_length)
            GPTJRotaryEmbedding.max_position_embedding = new_cache_length
        if layer_id == 0:
            GPTJRotaryEmbedding.position_ids = position_ids.transpose(0, 1).contiguous()
            GPTJRotaryEmbedding.sin = self.sin_cached[
                GPTJRotaryEmbedding.position_ids
            ].unsqueeze(2)
            GPTJRotaryEmbedding.cos = self.cos_cached[
                GPTJRotaryEmbedding.position_ids
            ].unsqueeze(2)
            if first_token and beam_size > 1:
                # 1st token
                # convert sin/cos from shape [seq, bs*beam, num_head, head_dim]
                # to shape [bs*beam, seq, num_head, head_dim]
                GPTJRotaryEmbedding.sin = GPTJRotaryEmbedding.sin.transpose(0, 1)
                GPTJRotaryEmbedding.cos = GPTJRotaryEmbedding.cos.transpose(0, 1)

        # 1st token
        # GPTJRotaryEmbedding.sin is in shape of [bs*beam, seq, num_head, head_dim]
        # 2nd to last token or greedy
        # GPTJRotaryEmbedding.sin is in shape of [seq, bs*beam, num_head, head_dim]
        return GPTJRotaryEmbedding.sin, GPTJRotaryEmbedding.cos

    def forward(self, query, key, position_ids, layer_id, beam_size, kv_seq_len):
        sin, cos = self.get_sin_cos(position_ids, layer_id, beam_size, kv_seq_len)
        if self.rotary_dim is not None:
            self.apply_rotary_pos_emb(
                query[:, :, :, : self.rotary_dim],
                key[:, :, :, : self.rotary_dim],
                sin,
                cos,
            )
        else:
            self.apply_rotary_pos_emb(query, key, sin, cos)
        return query, key


class LlamaRotaryEmbeddingBase(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbeddingBase):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbeddingBase):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaRotaryEmbeddingRef(torch.nn.Module):
    def __init__(self, config: IPEXTransformerConfig):
        super().__init__()
        self.config = config
        self.head_dim = int(config.embed_dim / config.num_attention_heads)
        self.max_position_embeddings = config.max_positions
        self.device = config.device
        self.dtype = config.dtype
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbeddingBase(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                device=self.device,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

        import os

        col_major = os.environ.get("COL_MAJOR", "OFF").upper() in [
            "1",
            "Y",
            "ON",
            "YES",
            "TRUE",
        ]
        self.row_major = not col_major

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        rotate_q = self.rotate_half(q)
        rotate_k = self.rotate_half(k)
        q_embed = (q * cos) + (rotate_q * sin)
        k_embed = (k * cos) + (rotate_k * sin)
        return q_embed, k_embed

    def forward(self, query, key, position_ids, layer_id, beam_size, kv_seq_len):
        # position_ids [bs*beam, seq_len]
        # cos [1, 1, kv_seq_len, head_dim]
        # sin [1, 1, kv_seq_len, head_dim]
        first_token = position_ids.shape[-1] > 1
        if self.row_major:
            if first_token and beam_size > 1:
                # from [bs*beam, seq, num_head, head_dim]
                # to [bs*beam, num_head, seq, head_dim]
                query_states = query.permute(0, 2, 1, 3).contiguous()
                key_states = key.permute(0, 2, 1, 3).contiguous()
            else:
                # from [seq, bs*beam, num_head, head_dim]
                # to [bs*beam, num_head, seq, head_dim]
                query_states = query.permute(1, 2, 0, 3).contiguous()
                key_states = key.permute(1, 2, 0, 3).contiguous()
        else:
            # from [bs*beam, seq, num_head, head_dim]
            # to [bs*beam, num_head, seq, head_dim]
            query_states = query.permute(0, 2, 1, 3).contiguous()
            key_states = key.permute(0, 2, 1, 3).contiguous()

        cos, sin = self.rotary_emb(key_states, seq_len=kv_seq_len)
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if self.row_major:
            if first_token and beam_size > 1:
                # frpm [bs*beam, num_head, seq, head_dim]
                # to [bs*beam, seq, num_head, head_dim]
                query_states = query_states.permute(0, 2, 1, 3).contiguous()
                key_states = key_states.permute(0, 2, 1, 3).contiguous()
            else:
                # from [bs*beam, num_head, seq, head_dim]
                # to [seq, bs*beam, num_head, head_dim]
                query_states = query_states.permute(2, 0, 1, 3).contiguous()
                key_states = key_states.permute(2, 0, 1, 3).contiguous()
        else:
            # frpm [bs*beam, num_head, seq, head_dim]
            # to [bs*beam, seq, num_head, head_dim]
            query_states = query_states.permute(0, 2, 1, 3).contiguous()
            key_states = key_states.permute(0, 2, 1, 3).contiguous()
        query.copy_(query_states)
        key.copy_(key_states)
        return query, key


class LlamaRotaryEmbedding(PositionalEmbedding):
    cos_cached = None
    sin_cached = None
    max_position_embedding = 2048

    def __init__(self, config: IPEXTransformerConfig, dtype):
        super().__init__(config, dtype)
        self.dim = int(config.embedding_dim / config.num_attention_head)
        LlamaRotaryEmbedding.max_position_embedding = config.max_positions
        self.base = config.positional_embedding_base
        self.device = config.device
        self.dtype = dtype
        if (
            LlamaRotaryEmbedding.sin_cached is None
            or LlamaRotaryEmbedding.cos_cached is None
        ):
            self.create_sin_cos_cache(LlamaRotaryEmbedding.max_position_embedding)

    def create_sin_cos_cache(self, pos):
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2).float().to(self.device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        device = self.inv_freq.device
        t = torch.arange(pos, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        LlamaRotaryEmbedding.cos_cached = emb.cos().to(self.dtype)
        LlamaRotaryEmbedding.sin_cached = emb.sin().to(self.dtype)

    def apply_rotary_pos_emb(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
    ):
        if query.shape == key.shape:
            cos = cos.expand(query.shape)
            sin = sin.expand(query.shape)
            torch.ops.torch_ipex.apply_rotary_embedding_half_qk(
                query, key, sin, cos, query, key
            )
        else:
            cos_q = cos.expand(query.shape)
            sin_q = sin.expand(query.shape)
            torch.ops.torch_ipex.apply_rotary_embedding_half(query, sin_q, cos_q, query)
            cos_k = cos.expand(key.shape)
            sin_k = sin.expand(key.shape)
            torch.ops.torch_ipex.apply_rotary_embedding_half(key, sin_k, cos_k, key)

    def apply_rotary_pos_emb_ref(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
    ):
        cos = cos.expand(query.shape).contiguous()
        sin = sin.expand(query.shape).contiguous()
        rotate_q = self.rotate_half(query)
        rotate_k = self.rotate_half(key)
        q_embed = (query * cos) + (rotate_q * sin)
        k_embed = (key * cos) + (rotate_k * sin)
        query.copy_(q_embed)
        key.copy_(k_embed)

    def get_sin_cos(self, position_ids, layer_id, beam_size, kv_seq_len, cache_format):
        if kv_seq_len >= LlamaRotaryEmbedding.max_position_embedding:
            new_cache_length = kv_seq_len + self.dynamic_cache_stride
            self.create_sin_cos_cache(new_cache_length)
            LlamaRotaryEmbedding.max_position_embedding = new_cache_length

        if layer_id == 0:
            LlamaRotaryEmbedding.position_ids = position_ids
            LlamaRotaryEmbedding.sin = self.sin_cached[
                LlamaRotaryEmbedding.position_ids
            ].unsqueeze(2)
            LlamaRotaryEmbedding.cos = self.cos_cached[
                LlamaRotaryEmbedding.position_ids
            ].unsqueeze(2)
            if cache_format == CacheFormat.FBNH:
                LlamaRotaryEmbedding.sin = LlamaRotaryEmbedding.sin.permute(1, 0, 2, 3)
                LlamaRotaryEmbedding.cos = LlamaRotaryEmbedding.cos.permute(1, 0, 2, 3)

            LlamaRotaryEmbedding.sin = LlamaRotaryEmbedding.sin.contiguous()
            LlamaRotaryEmbedding.cos = LlamaRotaryEmbedding.cos.contiguous()

        # 1st token
        # LlamaRotaryEmbedding.sin is in shape of [bs*beam, seq, 1, head_dim]
        # 2nd to last token or greedy
        # GPTJRotaryEmbedding.sin is in shape of [seq, bs*beam, 1, head_dim]
        return LlamaRotaryEmbedding.sin, LlamaRotaryEmbedding.cos

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        query,
        key,
        position_ids,
        layer_id,
        beam_size,
        kv_seq_len,
        cache_format=CacheFormat.BFNH,
    ):
        sin, cos = self.get_sin_cos(
            position_ids, layer_id, beam_size, kv_seq_len, cache_format
        )
        self.apply_rotary_pos_emb(query, key, sin, cos)
        return query, key


def apply_rotary_pos_emb(query, key, rotary_pos_emb_list):
    cur_len = query.shape[1]
    if len(rotary_pos_emb_list) == 1:
        rotary_pos_emb = rotary_pos_emb_list[0]
        rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
        rotary_pos_emb = (rotary_pos_emb,) * 2
        q_pos_emb, k_pos_emb = rotary_pos_emb
        # Slice the pos emb for current inference
        torch.ops.torch_ipex.apply_rotary_embedding_half(
            query, q_pos_emb[1], q_pos_emb[0], query
        )
        torch.ops.torch_ipex.apply_rotary_embedding_half(
            key, k_pos_emb[1], k_pos_emb[0], key
        )
    else:
        query_list = []
        key_list = []
        for i, rotary_pos_emb in enumerate(rotary_pos_emb_list):
            rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
            rotary_pos_emb = (rotary_pos_emb,) * 2
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # Slice the pos emb for current inference
            q_i, k_i = query[i : i + 1, :, :], key[i : i + 1, :, :]
            torch.ops.torch_ipex.apply_rotary_embedding_half(
                q_i, q_pos_emb[1], q_pos_emb[0], q_i
            )
            torch.ops.torch_ipex.apply_rotary_embedding_half(
                k_i, k_pos_emb[1], k_pos_emb[0], k_i
            )
            query_list += [q_i]
            key_list += [k_i]
        query = torch.cat(query_list, dim=0)
        key = torch.cat(key_list, dim=0)
    return query, key


class ChatGLMRotaryEmbedding(PositionalEmbedding):
    def __init__(self, config: IPEXTransformerConfig, dtype):
        super().__init__(config, dtype)

    def apply_rotary_pos_emb(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ):
        sin = rotary_pos_emb[..., 1]
        cos = rotary_pos_emb[..., 0]

        if query.shape == key.shape:
            cos = cos.expand(query.shape)
            sin = sin.expand(query.shape)
            torch.ops.torch_ipex.apply_rotary_embedding_two_qk(
                query, key, sin, cos, query, key
            )
        else:
            cos_q = cos.expand(query.shape)
            sin_q = sin.expand(query.shape)
            torch.ops.torch_ipex.apply_rotary_embedding_two(query, sin_q, cos_q, query)
            cos_k = cos.expand(key.shape)
            sin_k = sin.expand(key.shape)
            torch.ops.torch_ipex.apply_rotary_embedding_two(key, sin_k, cos_k, key)

    def apply_rotary_pos_emb_ref(
        self, x: torch.Tensor, rope_cache: torch.Tensor
    ) -> torch.Tensor:
        # x: [sq, b, np, hn]
        sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
        rot_dim = rope_cache.shape[-2] * 2
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        # truncate to support variable sizes
        rope_cache = rope_cache[:sq]
        xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
        rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out2 = x_out2.flatten(3)
        return torch.cat((x_out2, x_pass), dim=-1)

    def forward(self, query, key, rotary_pos_emb):
        rot_dim = rotary_pos_emb.shape[-2]
        self.apply_rotary_pos_emb(
            query[..., :rot_dim], key[..., :rot_dim], rotary_pos_emb
        )
        return query, key


class Phi3RotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config: IPEXTransformerConfig, dtype):
        super().__init__(config, dtype)


class Qwen2RotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config: IPEXTransformerConfig, dtype):
        super().__init__(config, dtype)
