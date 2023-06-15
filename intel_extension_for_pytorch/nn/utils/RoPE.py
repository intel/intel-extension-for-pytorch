import torch

class GPTJRotaryEmbedding(torch.nn.Module):
    sin = None
    cos = None
    position_ids = None

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, dtype=torch.float32):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float, device=device)
        sinusoid_inp = torch.einsum("i , j -> i j", t, inv_freq).float()
        embed_positions = torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)

        sin, cos = torch.split(embed_positions, embed_positions.shape[-1] // 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, 1).to(dtype)
        cos = torch.repeat_interleave(cos, 2, 1).to(dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def rotate_every_two(self, x: torch.Tensor) -> torch.Tensor:
        # the original rotary_every_two funtion used in the model
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

    def apply_rotary_pos_emb(self, query, key, sin, cos) -> torch.Tensor:
        torch.ops.torch_ipex.apply_rotary_embedding_qk(query, key, sin, cos, query, key)

    def get_sin_cos(self, position_ids):
        if GPTJRotaryEmbedding.position_ids is None or GPTJRotaryEmbedding.position_ids is not position_ids:
            GPTJRotaryEmbedding.sin = self.sin_cached[position_ids].unsqueeze(2)
            GPTJRotaryEmbedding.cos = self.cos_cached[position_ids].unsqueeze(2)
            GPTJRotaryEmbedding.position_ids = position_ids
        return GPTJRotaryEmbedding.sin, GPTJRotaryEmbedding.cos

    def forward(self, query, key, position_ids, rotary_dim=None):
        sin, cos = self.get_sin_cos(position_ids)
        if rotary_dim is not None:
            self.apply_rotary_pos_emb(query[:, :, :, : rotary_dim], key[:, :, :, : rotary_dim], sin, cos)
        else:
            self.apply_rotary_pos_emb(query, key, sin, cos)
        return query, key

class LlamaRotaryEmbedding(torch.nn.Module):
    sin = None
    cos = None
    position_ids = None
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, dtype=torch.float32):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        tmp = self.rotate_half(tensor) * sin
        tensor *= cos
        tensor += tmp

    def get_sin_cos(self, position_ids):
        if LlamaRotaryEmbedding.position_ids is None or LlamaRotaryEmbedding.position_ids is not position_ids:
            LlamaRotaryEmbedding.sin = self.sin_cached[position_ids].unsqueeze(2)
            LlamaRotaryEmbedding.cos = self.cos_cached[position_ids].unsqueeze(2)
            LlamaRotaryEmbedding.position_ids = position_ids
        return LlamaRotaryEmbedding.sin, LlamaRotaryEmbedding.cos
    
    def forward(self, key, value, position_ids, rotary_dim=None):
        sin, cos = self.get_sin_cos(position_ids)
        if rotary_dim is not None:
            self.apply_rotary_pos_emb(key[:, :, :, : rotary_dim], sin, cos)
            self.apply_rotary_pos_emb(query[:, :, :, : rotary_dim], sin, cos)
        else:
            self.apply_rotary_pos_emb(key, sin, cos)
            self.apply_rotary_pos_emb(query, sin, cos)
        return query, key
