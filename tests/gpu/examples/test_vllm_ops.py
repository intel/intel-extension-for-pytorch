import intel_extension_for_pytorch as ipex  # noqa F401
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from typing import Type, Optional, Union, Tuple


DTYPES = [torch.half, torch.bfloat16, torch.float]

NUM_TOKENS = [7, 2046]  # Arbitrary values for testing

D = [512, 5120]  # Arbitrary values for testing

SEEDS = [0]
DEVICES = ["xpu"]


class SiluAndMul(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        return torch.ops.torch_ipex.silu_and_mul(x, out)

    def naive_forward(self, x: torch.Tensor):
        dtype = x.dtype
        d = x.shape[-1] // 2
        x = x.to(torch.float32)
        ret = F.silu(x[..., :d]) * x[..., d:]
        return ret.to(dtype)


class GeluAndMul(nn.Module):
    def __init__(self, approximate: str) -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        return torch.ops.torch_ipex.gelu_and_mul(x, out, self.approximate)

    def naive_forward(self, x: torch.Tensor):
        dtype = x.dtype
        d = x.shape[-1] // 2
        x = x.to(torch.float32)
        ret = F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]
        return ret.to(dtype)


class QuickGELU(nn.Module):

    # https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        dtype = x.dtype
        x = x.to(torch.float32)
        ret = x * torch.sigmoid(1.702 * x)
        return ret.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = torch.empty_like(x)
        out = torch.ops.torch_ipex.gelu_quick(x)
        return out


def _apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1)
    )
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(
        1, 2
    )
    return x_out


class RotaryEmbedding(nn.Module):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        cache = self._compute_cos_sin_cache()
        self.use_native2 = False
        if not self.use_native2:
            cache = cache.to(dtype)
            self.register_buffer("cos_sin_cache", cache, persistent=False)
        else:
            cos, sin = cache.chunk(2, dim=-1)
            freqs_cis = cos + 1j * sin
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation equivalent to forward().

        This method mimics the implementation of the custom CUDA kernel
        used in `forward_cuda()`.
        """
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)

        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(
            positions.device, dtype=query.dtype
        )
        cos_sin = self.cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = (
            query_rot.float() * cos.float() + rotate_fn(query_rot).float() * sin.float()
        ).to(self.dtype)
        key_rot = (
            key_rot.float() * cos.float() + rotate_fn(key_rot).float() * sin.float()
        ).to(self.dtype)

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        query = query.flatten(-2)
        key = key.flatten(-2)
        return query, key

    def forward_native2(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Another PyTorch-native implementation of forward().

        This method might perform better than `forward_native()` when compiled.
        """
        if positions.dim() == 1:
            batch_size = 1
            seq_len = positions.shape[0]
        else:
            batch_size, seq_len = positions.shape
        if offsets is not None:
            positions = positions + offsets
        freqs_cis = self.freqs_cis.index_select(0, positions.flatten())
        freqs_cis = freqs_cis.view(batch_size, 1, seq_len, -1)

        query_shape = query.shape
        query = query.view(batch_size, seq_len, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = _apply_rotary_emb(query_rot, freqs_cis)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(batch_size, seq_len, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = _apply_rotary_emb(key_rot, freqs_cis)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self.cos_sin_cache = self.cos_sin_cache.to(positions.device, dtype=query.dtype)
        if offsets is not None:
            torch.ops.torch_ipex.rotary_embedding_batched(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
                self.rotary_dim,
                offsets,
            )
        else:
            torch.ops.torch_ipex.rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
                self.rotary_dim,
            )
        return query, key


@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_act_and_mul(
    activation: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)
    if activation == "silu":
        op = SiluAndMul()
    elif activation == "gelu":
        op = GeluAndMul(approximate="none")
    elif activation == "gelu_tanh":
        op = GeluAndMul(approximate="tanh")
    out = op(x)
    ref_out = op.naive_forward(x)

    # The SiLU and GELU implementations are equivalent to the native PyTorch
    # implementations, so we can do exact comparison.
    assert torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("activation", [QuickGELU])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_activation(
    activation: Type[torch.nn.Module],
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, d, dtype=dtype)
    layer = activation()
    out = layer(x)
    ref_out = layer.forward_native(x)
    assert torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3)


IS_NEOX_STYLE = [True, False]
DTYPES = [torch.half, torch.bfloat16, torch.float]
HEAD_SIZES = [128, 192, 256]
ROTARY_DIMS = [None, 32]  # None means rotary dim == head size
NUM_HEADS = [7, 17]  # Arbitrary values for testing
BATCH_SIZES = [1, 5]  # Arbitrary values for testing
SEQ_LENS = [11, 8192]  # Arbitrary values for testing
SEEDS = [0]
DEVICES = ["xpu"]


def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_rotary_embedding(
    is_neox_style: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    if rotary_dim is None:
        rotary_dim = head_size
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope = RotaryEmbedding(
        head_size, rotary_dim, max_position, base, is_neox_style, dtype
    )
    rope = rope.to(dtype=dtype)

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query = torch.randn(batch_size, seq_len, num_heads * head_size, dtype=dtype)
    key = torch.randn_like(query)

    ref_query, ref_key = rope.forward_native(positions, query, key)
    out_query, out_key = rope.forward(positions, query, key)
    # Compare the results.
    assert torch.allclose(out_query, ref_query, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_key, ref_key, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_batched_rotary_embedding(
    is_neox_style: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope = RotaryEmbedding(
        head_size, rotary_dim, max_position, base, is_neox_style, dtype
    )
    rope = rope.to(dtype=dtype)

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query = torch.randn(batch_size, seq_len, num_heads * head_size, dtype=dtype)
    key = torch.randn_like(query)

    ref_query, ref_key = rope.forward_native(positions, query, key)
    out_query, out_key = rope.forward(
        positions,
        query,
        key,
        offsets=torch.zeros(batch_size * seq_len, dtype=torch.long, device=device),
    )
    # Compare the results.
    assert torch.allclose(out_query, ref_query, atol=1e-3, rtol=1e-3)
    assert torch.allclose(out_key, ref_key, atol=1e-3, rtol=1e-3)
