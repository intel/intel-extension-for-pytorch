import random
import torch
import pytest
from typing import List, Tuple
import intel_extension_for_pytorch as ipex  # noqa


DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_LAYERS = [1]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# HEAD_SIZES = [64]
BLOCK_SIZES = [8, 16, 32]
# BLOCK_SIZES = [8]
NUM_BLOCKS = [1024, 3600]  # Arbitrary values for testing
# NUM_BLOCKS = [1024]
NUM_MAPPINGS = [256]  # Arbitrary values for testing
SEEDS = [0]


def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=dtype)
        key_cache.uniform_(-scale, scale)
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape, dtype=dtype)
        value_cache.uniform_(-scale, scale)
        value_caches.append(value_cache)
    return key_caches, value_caches


@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_copy_blocks(
    num_mappings: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str = "xpu",
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    # Generate random block mappings where each source block is mapped to two
    # destination blocks.
    assert 2 * num_mappings <= num_blocks
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remainig_blocks = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remainig_blocks, 2 * num_mappings)
    block_mapping = {}
    for i in range(num_mappings):
        src = src_blocks[i]
        dst1 = dst_blocks[2 * i]
        dst2 = dst_blocks[2 * i + 1]
        block_mapping[src] = [dst1, dst2]

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(
        num_blocks, block_size, num_layers, num_heads, head_size, dtype, seed, device
    )

    # Clone the KV caches.
    cloned_key_caches = [key_cache.clone() for key_cache in key_caches]
    cloned_value_caches = [value_cache.clone() for value_cache in value_caches]

    # Call the copy blocks kernel.
    torch.xpu.copy_blocks(key_caches, value_caches, block_mapping)

    # Run the reference implementation.
    for src, dsts in block_mapping.items():
        for dst in dsts:
            for cloned_key_cache in cloned_key_caches:
                cloned_key_cache[dst].copy_(cloned_key_cache[src])
            for cloned_value_cache in cloned_value_caches:
                cloned_value_cache[dst].copy_(cloned_value_cache[src])

    # Compare the results.
    for key_cache, cloned_key_cache in zip(key_caches, cloned_key_caches):
        assert torch.allclose(key_cache, cloned_key_cache, atol=1e-2, rtol=1e-2)
    for value_cache, cloned_value_cache in zip(value_caches, cloned_value_caches):
        assert torch.allclose(value_cache, cloned_value_cache, atol=1e-2, rtol=1e-2)
    torch.set_default_device("cpu")
    torch.xpu.empty_cache()


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_reshape_and_cache(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str = "xpu",
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long)

    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype)
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(
        num_blocks, block_size, 1, num_heads, head_size, dtype, seed, device
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Clone the KV caches.
    cloned_key_cache = key_cache.clone()
    cloned_value_cache = value_cache.clone()

    # Call the reshape_and_cache kernel.
    torch.xpu.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    # Run the reference implementation.
    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies[i]
        block_offset = block_offsets[i]
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    assert torch.allclose(key_cache, cloned_key_cache, atol=1e-2, rtol=1e-2)
    assert torch.allclose(value_cache, cloned_value_cache, atol=1e-2, rtol=1e-2)
    torch.set_default_device("cpu")
    torch.xpu.empty_cache()
