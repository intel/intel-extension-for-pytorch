import random
import torch
import pytest
from typing import List, Tuple
from enum import Enum
import intel_extension_for_pytorch as ipex  # noqa

COPYING_DIRECTION = [("xpu", "cpu"), ("xpu", "xpu"), ("cpu", "xpu")]
DTYPES = [torch.half]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_LAYERS = [1]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 96, 112]
# HEAD_SIZES = [64]
BLOCK_SIZES = [16, 32]
# BLOCK_SIZES = [8]
NUM_BLOCKS = [1024, 3600]  # Arbitrary values for testing
# NUM_BLOCKS = [1024]
NUM_MAPPINGS = [256]  # Arbitrary values for testing
SEEDS = [0]


class KVCacheFormat(Enum):
    Paged = 0
    Chunked = 1


CACHE_FORMAT = [KVCacheFormat.Paged, KVCacheFormat.Chunked]


def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    cache_format: KVCacheFormat = KVCacheFormat.Paged,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    if cache_format == KVCacheFormat.Chunked:
        key_cache_shape = (num_blocks, block_size, num_heads, head_size)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=dtype)
        key_cache.uniform_(-scale, scale)
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    if cache_format == KVCacheFormat.Chunked:
        value_cache_shape = (num_blocks, block_size, num_heads, head_size)
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
    ipex.llm.modules.PagedAttention.copy_blocks(key_caches, value_caches, block_mapping)

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


@pytest.mark.parametrize("direction", COPYING_DIRECTION)
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_swap_blocks(
    direction: Tuple[str, str],
    num_mappings: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.xpu.manual_seed(seed)
    device = torch.device("xpu")
    src_device = device if direction[0] == "xpu" else "cpu"
    dst_device = device if direction[1] == "xpu" else "cpu"

    src_blocks = random.sample(range(num_blocks), num_mappings)
    # For the same device, mapping must not overlap
    if src_device == dst_device:
        remaining_blocks = list(set(range(num_blocks)) - set(src_blocks))
        dst_blocks = random.sample(remaining_blocks, num_mappings)
    else:
        dst_blocks = random.sample(range(num_blocks), num_mappings)

    block_mapping = list(zip(src_blocks, dst_blocks))
    block_mapping_tensor = torch.tensor(
        block_mapping, dtype=torch.int64, device="cpu"
    ).view(-1, 2)

    # Create the KV caches on the first device.
    src_key_caches, src_value_caches = kv_cache_factory(
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        dtype,
        seed,
        src_device,
    )

    # Create the KV caches on the second device.
    dist_key_caches, dist_value_caches = kv_cache_factory(
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        dtype,
        seed,
        dst_device,
    )

    src_key_caches_clone = src_key_caches[0].clone()
    src_value_caches_clone = src_value_caches[0].clone()

    # Call the swap_blocks kernel.
    ipex.llm.modules.PagedAttention.swap_blocks(
        src_key_caches[0], dist_key_caches[0], block_mapping_tensor
    )
    ipex.llm.modules.PagedAttention.swap_blocks(
        src_value_caches[0], dist_value_caches[0], block_mapping_tensor
    )

    for src, dst in block_mapping:
        assert torch.allclose(
            src_key_caches_clone[src].cpu(), dist_key_caches[0][dst].cpu()
        )
        assert torch.allclose(
            src_value_caches_clone[src].cpu(), dist_value_caches[0][dst].cpu()
        )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cache_format", CACHE_FORMAT)
@torch.inference_mode()
def test_reshape_and_cache(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    cache_format: KVCacheFormat,
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
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        dtype,
        seed,
        device,
        cache_format,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Clone the KV caches.
    cloned_key_cache = key_cache.clone()
    cloned_value_cache = value_cache.clone()

    cloned_key_cache_1 = key_cache.clone()
    cloned_value_cache_1 = value_cache.clone()

    # Call the reshape_and_cache kernel.
    if cache_format == KVCacheFormat.Paged:
        ipex.llm.modules.PagedAttention.reshape_and_cache(
            key, value, cloned_key_cache_1, cloned_value_cache_1, slot_mapping
        )
        ipex.llm.modules.PagedAttention.reshape_and_cache(
            key, value, key_cache, value_cache, slot_mapping.to(torch.int32)
        )
    else:
        ipex.llm.modules.PagedAttention.reshape_and_cache_flash(
            key, value, cloned_key_cache_1, cloned_value_cache_1, slot_mapping
        )
        ipex.llm.modules.PagedAttention.reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping.to(torch.int32)
        )

    # Run the reference implementation.
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets = block_offsets.cpu().tolist()
    if cache_format == KVCacheFormat.Chunked:
        for i in range(num_tokens):
            block_idx = block_indicies[i]
            block_offset = block_offsets[i]
            cloned_key_cache[block_idx, block_offset, :, :] = key[i]
            cloned_value_cache[block_idx, block_offset, :, :] = value[i]
    else:
        reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
        for i in range(num_tokens):
            block_idx = block_indicies[i]
            block_offset = block_offsets[i]
            cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
            cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    assert torch.allclose(key_cache, cloned_key_cache, atol=1e-2, rtol=1e-2)
    assert torch.allclose(value_cache, cloned_value_cache, atol=1e-2, rtol=1e-2)
    assert torch.allclose(cloned_key_cache_1, cloned_key_cache, atol=1e-2, rtol=1e-2)
    assert torch.allclose(
        cloned_value_cache_1, cloned_value_cache, atol=1e-2, rtol=1e-2
    )
    torch.set_default_device("cpu")
    torch.xpu.empty_cache()


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cache_format", CACHE_FORMAT)
@pytest.mark.parametrize("qtype", [torch.float8_e5m2, torch.float8_e4m3fn])
@torch.inference_mode()
def test_reshape_and_cache_fp8(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    cache_format: KVCacheFormat,
    qtype: torch.dtype,
    device: str = "xpu",
    # qtype: torch.dtype = torch.float8_e5m2,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.set_default_device(device)

    if qtype == torch.float8_e5m2:
        k_scale = 1.0
        v_scale = 1.0
        qtype_str = "fp8_e5m2"
    else:
        k_scale = 2.0
        v_scale = 2.0
        qtype_str = "fp8_e4m3"

    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long)

    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype)
    _, key, value = qkv.unbind(dim=1)
    key = key.to(qtype).to(dtype)
    value = value.to(qtype).to(dtype)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        dtype,
        seed,
        "cpu",
        cache_format,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    key_cache = key_cache.to(qtype).to(dtype)
    value_cache = value_cache.to(qtype).to(dtype)

    key_cache_xpu = key_cache.to(device)
    value_cache_xpu = value_cache.to(device)

    # Clone the KV caches.
    cloned_key_cache = key_cache.clone().to(device)
    cloned_value_cache = value_cache.clone().to(device)
    key_cache_fp8 = key_cache.to(qtype).to(device)
    value_cache_fp8 = value_cache.to(qtype).to(device)

    cloned_key_cache_1 = key_cache.clone().to(device)
    cloned_value_cache_1 = value_cache.clone().to(device)
    key_cache_fp8_1 = key_cache.to(qtype).to(device)
    value_cache_fp8_1 = value_cache.to(qtype).to(device)
    assert torch.allclose(
        key_cache_fp8_1.to(dtype), key_cache_xpu, atol=1e-2, rtol=1e-2
    )

    key = key.to(device)
    value = value.to(device)

    # Call the reshape_and_cache kernel.
    if cache_format == KVCacheFormat.Paged:
        # print(key)
        # print(key_cache_fp8_1)
        ipex.llm.modules.PagedAttention.reshape_and_cache(
            key,
            value,
            key_cache_fp8_1,
            value_cache_fp8_1,
            slot_mapping,
            qtype_str,
            k_scale,
            v_scale,
        )
        ipex.llm.modules.PagedAttention.reshape_and_cache(
            key,
            value,
            key_cache_fp8,
            value_cache_fp8,
            slot_mapping.to(torch.int32),
            qtype_str,
            k_scale,
            v_scale,
        )
    else:
        ipex.llm.modules.PagedAttention.reshape_and_cache_flash(
            key,
            value,
            key_cache_fp8_1,
            value_cache_fp8_1,
            slot_mapping,
            qtype_str,
            k_scale,
            v_scale,
        )
        ipex.llm.modules.PagedAttention.reshape_and_cache_flash(
            key,
            value,
            key_cache_fp8,
            value_cache_fp8,
            slot_mapping.to(torch.int32),
            qtype_str,
            k_scale,
            v_scale,
        )

    # Run the reference implementation.
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets = block_offsets.cpu().tolist()
    if cache_format == KVCacheFormat.Chunked:
        for i in range(num_tokens):
            block_idx = block_indicies[i]
            block_offset = block_offsets[i]
            cloned_key_cache[block_idx, block_offset, :, :] = key[i] * k_scale
            cloned_value_cache[block_idx, block_offset, :, :] = value[i] * v_scale
    else:
        reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
        for i in range(num_tokens):
            block_idx = block_indicies[i]
            block_offset = block_offsets[i]
            cloned_key_cache[block_idx, :, :, block_offset, :] = (
                reshaped_key[i] * k_scale
            )
            cloned_value_cache[block_idx, :, :, block_offset] = value[i] * v_scale

    assert torch.allclose(
        key_cache_fp8_1.to(dtype), cloned_key_cache, atol=1e-2, rtol=1e-2
    )
    assert torch.allclose(
        value_cache_fp8_1.to(dtype),
        cloned_value_cache,
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.allclose(
        key_cache_fp8.to(dtype), cloned_key_cache, atol=1e-2, rtol=1e-2
    )
    assert torch.allclose(
        value_cache_fp8.to(dtype),
        cloned_value_cache,
        atol=1e-2,
        rtol=1e-2,
    )

    torch.set_default_device("cpu")
    torch.xpu.empty_cache()
