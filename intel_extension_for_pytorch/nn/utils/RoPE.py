import torch
import torch.nn as nn
from ._transformer_configuration import IPEXTransformerConfig

class PositionalEmbedding(nn.Module):
    def __init__(self, 
                 config: IPEXTransformerConfig):
        super().__init__()
        self.config = config

    def forward(self, query, key, position_ids):
        return query, key


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class GPTJRotaryEmbedding(PositionalEmbedding):
    sin = None
    cos = None
    position_ids = None

    def __init__(self,
                 config: IPEXTransformerConfig):
        super().__init__(config=config)
        self.rotary_dim = config.rotary_dim
        self.max_position_embedding = config.max_positions
        self.base = config.positional_embedding_base
        self.device = config.device
        self.dtype = config.dtype
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2).float().to(self.device) / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(self.max_position_embedding, dtype=torch.float, device=self.device)
        sinusoid_inp = torch.einsum("i , j -> i j", t, inv_freq).float()
        embed_positions = torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)

        sin, cos = torch.split(embed_positions, embed_positions.shape[-1] // 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, 1).to(self.dtype).to(self.device)
        cos = torch.repeat_interleave(cos, 2, 1).to(self.dtype).to(self.device)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def rotate_every_two(self, x: torch.Tensor) -> torch.Tensor:
        # the original rotary_every_two funtion used in the model
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

    def apply_rotary_pos_emb(self, query, key, sin, cos) -> torch.Tensor:
        torch.ops.torch_ipex.apply_rotary_embedding_two(query, key, sin, cos, query, key)

    def get_sin_cos(self, position_ids):
        if GPTJRotaryEmbedding.position_ids is None or GPTJRotaryEmbedding.position_ids is not position_ids:
            GPTJRotaryEmbedding.sin = self.sin_cached[position_ids].unsqueeze(2)
            GPTJRotaryEmbedding.cos = self.cos_cached[position_ids].unsqueeze(2)
            GPTJRotaryEmbedding.position_ids = position_ids
        return GPTJRotaryEmbedding.sin, GPTJRotaryEmbedding.cos

    def forward(self, query, key, position_ids):
        sin, cos = self.get_sin_cos(position_ids)
        if self.rotary_dim is not None:
            self.apply_rotary_pos_emb(query[:, :, :, : self.rotary_dim], key[:, :, :, : self.rotary_dim], sin, cos)
        else:
            self.apply_rotary_pos_emb(query, key, sin, cos)
        return query, key

class LlamaRotaryEmbedding(torch.nn.Module):
    sin = None
    cos = None
    position_ids = None
    def __init__(self,
                 config: IPEXTransformerConfig):
        super().__init__()
        self.dim = int(config.embed_dim / config.num_attention_heads)
        self.max_position_embedding = config.max_positions
        self.base = config.positional_embedding_base
        self.device = config.device
        self.dtype = config.dtype
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = self.max_position_embedding
        t = torch.arange(self.max_seq_len_cached, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        print(emb.size())
        self.register_buffer("cos_cached", emb.cos().to(self.dtype).to(self.device), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(self.dtype).to(self.device), persistent=False)

    def apply_rotary_pos_emb(self, query: torch.Tensor, key: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        torch.ops.torch_ipex.apply_rotary_embedding_half(query, key, sin, cos, query, key)

    def get_sin_cos(self, position_ids):
        if LlamaRotaryEmbedding.position_ids is None or LlamaRotaryEmbedding.position_ids is not position_ids:
            LlamaRotaryEmbedding.sin = self.sin_cached[position_ids].unsqueeze(2)
            LlamaRotaryEmbedding.cos = self.cos_cached[position_ids].unsqueeze(2)
            LlamaRotaryEmbedding.position_ids = position_ids
        return LlamaRotaryEmbedding.sin, LlamaRotaryEmbedding.cos

    def forward(self, query, key, position_ids):
        kv_len = key.size(1)
        sin, cos = self.get_sin_cos(kv_len, position_ids)
        self.apply_rotary_pos_emb(query, key, sin, cos)
        return query, key
