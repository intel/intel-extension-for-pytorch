import torch
import random
from typing import List, Optional, Tuple
import intel_extension_for_pytorch as ipex  # noqa
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
import pytest

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = 1024
NUM_BLOCKS = 128  # Arbitrary values for testing
PARTITION_SIZE = 512

DTYPES = [torch.float16]
NUM_GEN_SEQS = [1]  # Arbitrary values for testing
NUM_HEADS = [1]
HEAD_SIZES = [64]
BLOCK_SIZES = [32]
USE_ALIBI = [False]
SEEDS = [0]


class TestChunkedPrefill(TestCase):
    def ref_masked_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = query.float()
        key = key.float()
        value = value.float()

        attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key)
        if attn_mask is not None:
            attn_mask = attn_mask.float()
            attn_weights = attn_weights + attn_mask
        attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
        out = torch.einsum("hqk,khd->qhd", attn_weights, value)
        return out

    def ref_chunked_prefill(
        self,
        output: torch.Tensor,
        query: torch.Tensor,  # (num_tokens, num_heads, head_size)
        num_queries_per_kv: int,
        key_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_size)
        value_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_size,)
        block_tables: torch.Tensor,  # (num_seqs, max_num_blocks_per_seq)
        cu_seqlen_q: torch.Tensor,  # (num_seqs + 1,)
        cu_seqlen_k: torch.Tensor,  # (num_seqs + 1,)
        max_seqlen_q: int,
        max_seqlen_k: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        causal: bool = True,
    ) -> None:
        query = query.to("xpu")
        key_cache = key_cache.to("xpu")
        value_cache = value_cache.to("xpu")
        block_tables = block_tables.to("xpu")
        cu_seqlen_k = cu_seqlen_k.to("xpu")
        cu_seqlen_q = cu_seqlen_q.to("xpu")
        num_query_heads = query.shape[1]
        head_dim = value_cache.shape[3]
        num_kv_heads = value_cache.shape[2]
        block_size = value_cache.shape[1]
        num_batch = cu_seqlen_q.shape[0] - 1
        num_tokens = query.shape[0]
        max_num_blocks_per_seq = block_tables.shape[1]

        key = key_cache[block_tables].view(
            num_batch, max_num_blocks_per_seq * block_size, num_kv_heads, head_dim
        )

        value = value_cache[block_tables].view(
            num_batch, max_num_blocks_per_seq * block_size, num_kv_heads, head_dim
        )
        key = key[:, :max_seqlen_k, :, :]
        value = value[:, :max_seqlen_k, :, :]

        seqlen_k = cu_seqlen_k[1:] - cu_seqlen_k[:-1]
        seqlen_q = cu_seqlen_q[1:] - cu_seqlen_q[:-1]
        seqlen_q = seqlen_q.view(-1, 1)
        seqlen_k = seqlen_k.view(-1, 1)
        seqlen_diff = seqlen_k - seqlen_q
        q_idx_mask = (
            torch.arange(0, max_seqlen_q, device="xpu").view(1, -1).repeat(num_batch, 1)
        )
        k_idx_mask = (
            torch.arange(0, max_seqlen_k, device="xpu").view(1, -1).repeat(num_batch, 1)
        )
        q_mask = q_idx_mask < seqlen_q
        k_mask = k_idx_mask < seqlen_k

        # calculate idx for causal mask of query    [batch, max_seqlen_q]
        causal_mask_idx = (q_idx_mask + seqlen_diff)[q_mask]

        # generate causal mask [batch, max_seqlen_q, max_seqlen_k]
        tril_mask = torch.tril(torch.ones(max_seqlen_k, max_seqlen_k, device="xpu"))
        tril_mask[tril_mask == 0] = float("-inf")
        tril_mask[tril_mask == 1] = 0
        causal_mask = tril_mask[causal_mask_idx]
        causal_mask_padding = torch.empty(
            [num_batch, max_seqlen_q, max_seqlen_k], device="xpu"
        ).fill_(float("-inf"))
        causal_mask_padding[q_mask] = causal_mask
        # to [batch, num_heads, max_seqlen_q, max_seqlen_k]
        causal_mask_padding = causal_mask_padding.unsqueeze(1)

        pad_q = torch.zeros(
            [num_batch, max_seqlen_q, num_query_heads, head_dim],
            device="xpu",
            dtype=query.dtype,
        )
        pad_k = torch.zeros(
            [num_batch, max_seqlen_k, num_kv_heads, head_dim],
            device="xpu",
            dtype=key.dtype,
        )
        pad_v = torch.zeros(
            [num_batch, max_seqlen_k, num_kv_heads, head_dim],
            device="xpu",
            dtype=value.dtype,
        )
        pad_q[q_mask] = query
        pad_k[k_mask] = key[k_mask]
        pad_v[k_mask] = value[k_mask]

        if num_query_heads > num_kv_heads:
            pad_k = pad_k.view([num_batch, max_seqlen_k, num_kv_heads, 1, head_dim])
            pad_k = pad_k.repeat(1, 1, 1, num_query_heads // num_kv_heads, 1).view(
                [num_batch, max_seqlen_k, num_query_heads, head_dim]
            )
            pad_v = pad_v.view([num_batch, max_seqlen_k, num_kv_heads, 1, head_dim])
            pad_v = pad_v.repeat(1, 1, 1, num_query_heads // num_kv_heads, 1).view(
                [num_batch, max_seqlen_k, num_query_heads, head_dim]
            )
        # permute to [b, h, n, k]
        pad_q = pad_q.permute(0, 2, 1, 3)
        pad_k = pad_k.permute(0, 2, 1, 3)
        pad_v = pad_v.permute(0, 2, 1, 3)
        attn_mask = torch.empty([num_batch, 1, 1, max_seqlen_k], device="xpu").fill_(
            float("-inf")
        )
        attn_mask[:, :, :, :max_seqlen_k].masked_fill_(k_mask[:, None, None, :], 0)
        # [b, h, f, t]
        attn_weights = torch.einsum("bhqd,bhkd->bhqk", pad_q, pad_k)
        attn_weights *= scale
        attn_mask = attn_mask.float()
        attn_weights = attn_weights + attn_mask
        if causal:
            attn_weights = attn_weights + causal_mask_padding

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, pad_v.float())
        attn_output = attn_output.permute(0, 2, 1, 3)

        attn_output = (
            attn_output[q_mask].view([-1, num_query_heads, head_dim]).to(output.dtype)
        )
        output.copy_(attn_output)
        return attn_output

    def create_q_buffer(
        self, cu_seqlen_q, num_query_heads, head_size, dtype, init_value=0
    ):
        num_tokens = cu_seqlen_q[-1]
        query = torch.empty(
            num_tokens, num_query_heads, head_size, dtype=dtype, device="cpu"
        )
        if not init_value:
            query.uniform_(-1, 1)
        else:
            query.fill_(init_value)
        return query

    def create_kv_caches(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
        seed: int,
        init_value=0,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        torch.random.manual_seed(seed)
        torch.manual_seed(seed)

        scale = head_size**-0.5
        key_value_cache_shape = (num_blocks, block_size, num_heads, head_size)
        key_caches = []
        for _ in range(num_layers):
            key_cache = torch.empty(size=key_value_cache_shape, dtype=dtype)
            if not init_value:
                key_cache.uniform_(-scale, scale)
            else:
                key_cache.fill_(1)
            key_caches.append(key_cache)

        value_caches = []
        for _ in range(num_layers):
            value_cache = torch.empty(size=key_value_cache_shape, dtype=dtype)
            if not init_value:
                value_cache.uniform_(-scale, scale)
            else:
                value_cache.fill_(1)
            value_caches.append(value_cache)
        return key_caches, value_caches

    def chunk_prefill(
        self,
        num_seqs,
        max_seqlen,
        num_heads,
        head_size,
        block_size,
        use_alibi,
        is_causal,
        version,
        dtype,
    ) -> None:
        seed = 0

        random.seed(seed)
        torch.random.manual_seed(seed)
        torch.manual_seed(seed)

        scale = float(1.0 / (head_size**0.5))
        # TODO: support GQA
        num_query_heads, num_kv_heads = num_heads
        assert num_query_heads % num_kv_heads == 0
        num_queries_per_kv = num_query_heads // num_kv_heads
        alibi_slopes = None
        if use_alibi:
            alibi_slopes = torch.rand(
                num_seqs, max_seqlen, max_seqlen, device="cpu", dtype=dtype
            )
        context_lens = [random.randint(1, max_seqlen) for _ in range(num_seqs)]

        max_seqlen_k = max(context_lens)
        context_lens = [0] + context_lens
        context_lens = torch.tensor(context_lens, dtype=torch.int, device="cpu")

        # Create the block tables.NUM_PREFILL_SEQS
        max_num_blocks_per_seq = (max_seqlen_k + block_size - 1) // block_size
        block_tables = []
        for _ in range(num_seqs):
            block_table = [
                random.randint(0, max_num_blocks_per_seq - 1)
                for i in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        block_tables = torch.tensor(block_tables, dtype=torch.int, device="cpu")
        cu_seqlen_k = torch.cumsum(context_lens, 0)
        q_lens = context_lens[1:] if version == "chunked_prefill" else [1] * num_seqs
        q_lens = [random.randint(1, max_lens) for max_lens in q_lens]
        max_seqlen_q = max(q_lens)
        q_lens = [0] + q_lens
        q_lens_tensor = torch.tensor(q_lens, dtype=torch.int, device="cpu")
        cu_seqlen_q = torch.cumsum(q_lens_tensor, 0)

        query = self.create_q_buffer(cu_seqlen_q, num_query_heads, head_size, dtype)
        key_caches, value_caches = self.create_kv_caches(
            max_num_blocks_per_seq, block_size, 1, num_kv_heads, head_size, dtype, seed
        )
        key_cache, value_cache = key_caches[0], value_caches[0]
        # Call the paged attention kernel.
        output = torch.zeros_like(query)

        xpu_device = torch.device("xpu")
        cu_seqlen_q_xpu = cu_seqlen_q.to(xpu_device).int()
        cu_seqlen_k_xpu = cu_seqlen_k.to(xpu_device).int()
        output_xpu = output.to(xpu_device)
        query_xpu = query.to("xpu")
        key_cache_xpu = key_cache.to(xpu_device)
        value_cache_xpu = value_cache.to(xpu_device)
        block_tables_xpu = block_tables.to(xpu_device)

        alibi_slopes_xpu = None

        # execute ref path of chunked prefill
        output = output.to("xpu")
        self.ref_chunked_prefill(
            output,
            query,
            num_queries_per_kv,
            key_cache,
            value_cache,
            block_tables,
            cu_seqlen_q,
            cu_seqlen_k,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            alibi_slopes,
            is_causal,
        )

        ipex.llm.modules.PagedAttention.flash_attn_varlen_func(
            output_xpu,
            query_xpu,
            key_cache_xpu,
            value_cache_xpu,
            cu_seqlen_q_xpu,
            cu_seqlen_k_xpu,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            is_causal,
            block_tables_xpu,
            alibi_slopes_xpu,
        )

        torch.testing.assert_close(output.cpu(), output_xpu.cpu(), atol=3e-3, rtol=1e-3)

    # @parametrize("num_gen_seqs", [1, 3, 8, 13])
    @parametrize("num_gen_seqs", [1, 3, 8])
    @parametrize("max_seqlen_k", [8, 1024, 2088])
    # @parametrize("max_seqlen_k", [76])
    @parametrize("num_heads", [(16, 16)])
    @parametrize("head_size", [64, 70, 96, 128, 256])
    # @parametrize("head_size", [64])
    @parametrize("block_size", [16, 32, 64, 128])
    @parametrize("use_alibi", [False])
    @parametrize("is_causal", [False, True])
    @parametrize("dtype", [torch.float16])
    @pytest.mark.skipif(
        not torch.xpu.has_2d_block_array(),
        reason="have accuracy issue with compiler 2024.1 on ATSM, disable it as a WA for now",
    )
    def test_chunked_prefill(
        self,
        num_gen_seqs,
        max_seqlen_k,
        num_heads,
        head_size,
        block_size,
        use_alibi,
        is_causal,
        dtype,
    ):
        self.chunk_prefill(
            num_gen_seqs,
            max_seqlen_k,
            num_heads,
            head_size,
            block_size,
            use_alibi,
            is_causal,
            "chunked_prefill",
            dtype,
        )

    @parametrize("num_gen_seqs", [1, 3, 8])
    # @parametrize("num_gen_seqs", [13])
    @parametrize("max_seqlen_k", [8, 76, 512, 2088])
    # @parametrize("max_seqlen_k", [76])
    @parametrize("num_heads", [(16, 16)])
    @parametrize("head_size", [64, 70, 96, 128, 256])
    # @parametrize("head_size", [64])
    @parametrize("block_size", [16, 32, 64, 128])
    @parametrize("use_alibi", [False])
    @parametrize("is_causal", [False])
    @parametrize("dtype", [torch.float16])
    @pytest.mark.skipif(
        not torch.xpu.has_2d_block_array(),
        reason="have accuracy issue with compiler 2024.1 on ATSM, disable it as a WA for now",
    )
    def test_flash_decode(
        self,
        num_gen_seqs,
        max_seqlen_k,
        num_heads,
        head_size,
        block_size,
        use_alibi,
        is_causal,
        dtype,
    ):
        self.chunk_prefill(
            num_gen_seqs,
            max_seqlen_k,
            num_heads,
            head_size,
            block_size,
            use_alibi,
            is_causal,
            "flash_decoding",
            dtype,
        )


instantiate_parametrized_tests(TestChunkedPrefill)

if __name__ == "__main__":
    run_tests()
