import torch
import random
from typing import List, Optional, Tuple
import pytest
import intel_extension_for_pytorch as ipex  # noqa
from torch.testing._internal.common_utils import (
    TestCase,
)

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


class TestPagedAttention(TestCase):
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

    def ref_single_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        num_queries_per_kv: int,
        key_cache: torch.Tensor,  # (num_blocks, num_heads, head_size, block_size)
        value_cache: torch.Tensor,  # (num_blocks, num_heads, head_size, block_size)
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
    ) -> None:
        num_query_heads = query.shape[1]
        num_kv_heads = value_cache.shape[1]
        head_size = value_cache.shape[2]
        block_size = value_cache.shape[3]
        num_seqs = query.shape[0]

        block_tables = block_tables.cpu().tolist()
        context_lens = context_lens.cpu().tolist()
        for i in range(num_seqs):
            q = query[i].unsqueeze(0)
            block_table = block_tables[i]
            context_len = int(context_lens[i])

            keys = []
            values = []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = key_cache[block_number, :, :, block_offset]
                k = k.reshape(num_kv_heads, head_size)
                keys.append(k)

                v = value_cache[block_number, :, :, block_offset]
                values.append(v)
            keys = torch.stack(keys, dim=0)
            values = torch.stack(values, dim=0)
            if num_queries_per_kv > 1:
                # Handle MQA and GQA
                keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
                values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)
            alibi_bias = None
            if alibi_slopes is not None:
                # Create the ALiBi bias used in the paged attention kernel.
                position_ids = torch.arange(context_len, device="cpu").int()
                alibi_bias = (position_ids - context_len + 1).float()
                alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)

            out = self.ref_masked_attention(q, keys, values, scale, alibi_bias)
            out = out.view(num_query_heads, head_size)
            output[i].copy_(out, non_blocking=True)

    def create_q_buffer(
        self, num_seqs, num_query_heads, head_size, dtype, init_value=0
    ):
        query = torch.empty(
            num_seqs, num_query_heads, head_size, dtype=dtype, device="cpu"
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
        key_value_cache_shape = (num_blocks, num_heads, head_size, block_size)
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

    def paged_attention(self, version) -> None:
        num_seqs = 4
        num_heads = [16, 16]
        head_size = 256
        use_alibi = False
        block_size = 32
        dtype = torch.float16
        seed = 0

        random.seed(seed)
        torch.random.manual_seed(seed)
        torch.manual_seed(seed)

        scale = float(1.0 / (head_size**0.5))
        # TODO: support GQA
        num_query_heads, num_kv_heads = num_heads
        assert num_query_heads % num_kv_heads == 0
        num_queries_per_kv = num_query_heads // num_kv_heads
        head_mapping = torch.repeat_interleave(
            torch.arange(num_kv_heads, dtype=torch.int32, device="cpu"),
            num_queries_per_kv,
        )
        alibi_slopes = None
        if use_alibi:
            alibi_slopes = torch.randn(num_query_heads, dtype=torch.float, device="cpu")

        context_lens = [512 + i for i in range(num_seqs)]

        max_context_len = max(context_lens)
        context_lens = torch.tensor(context_lens, dtype=torch.int, device="cpu")

        # Create the block tables.NUM_PREFILL_SEQS
        max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        block_tables = []
        for _ in range(num_seqs):
            block_table = [
                random.randint(0, max_num_blocks_per_seq - 1)
                for i in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        block_tables = torch.tensor(block_tables, dtype=torch.int, device="cpu")

        # Create the KV caches.
        query = self.create_q_buffer(num_seqs, num_query_heads, head_size, dtype)
        key_caches, value_caches = self.create_kv_caches(
            max_num_blocks_per_seq, block_size, 1, num_kv_heads, head_size, dtype, seed
        )
        key_cache, value_cache = key_caches[0], value_caches[0]
        # Call the paged attention kernel.
        output = torch.empty_like(query)

        xpu_device = torch.device("xpu")
        output_xpu = output.to(xpu_device)
        output_xpu_clone = output_xpu.clone()
        query_xpu = query.to(xpu_device)
        key_cache_xpu = key_cache.to(xpu_device)
        value_cache_xpu = value_cache.to(xpu_device)
        head_mapping_xpu = head_mapping.to(xpu_device)
        block_tables_xpu = block_tables.to(xpu_device)
        context_lens_xpu = context_lens.to(xpu_device)
        if use_alibi:
            alibi_slopes_xpu = alibi_slopes.to(xpu_device)
        else:
            alibi_slopes_xpu = None

        if version == "v1":
            torch.xpu.paged_attention_v1(
                output_xpu,
                query_xpu,
                key_cache_xpu,
                value_cache_xpu,
                head_mapping_xpu,
                scale,
                block_tables_xpu,
                context_lens_xpu,
                block_size,
                max_context_len,
                alibi_slopes_xpu,
            )

            ipex.llm.modules.PagedAttention.single_query_cached_kv_attention(
                output_xpu_clone,
                query_xpu,
                key_cache_xpu,
                value_cache_xpu,
                head_mapping_xpu,
                scale,
                block_tables_xpu,
                context_lens_xpu,
                block_size,
                max_context_len,
                alibi_slopes,
            )
        elif version == "v2":
            num_partitions = (max_context_len + PARTITION_SIZE - 1) // PARTITION_SIZE
            # Note: PARTITION_SIZE must be equal to paged_attention_policy.hpp::paged_attention_policy_v2::partition_size
            assert PARTITION_SIZE == 512
            assert PARTITION_SIZE % block_size == 0
            num_seqs, num_heads, head_size = output_xpu.shape
            tmp_output_xpu = torch.empty(
                size=(num_seqs, num_heads, num_partitions, head_size),
                dtype=output_xpu.dtype,
                device=output_xpu.device,
            )
            exp_sums_xpu = torch.empty(
                size=(num_seqs, num_heads, num_partitions),
                dtype=torch.float32,
                device=output_xpu.device,
            )
            max_logits_xpu = torch.empty_like(exp_sums_xpu)
            torch.xpu.paged_attention_v2(
                output_xpu,
                exp_sums_xpu,
                max_logits_xpu,
                tmp_output_xpu,
                query_xpu,
                key_cache_xpu,
                value_cache_xpu,
                head_mapping_xpu,
                block_tables_xpu,
                context_lens_xpu,
                scale,
                block_size,
                max_context_len,
                alibi_slopes_xpu,
            )
        else:
            assert False, f"Unknown version: {version}"  # noqa

        # Run the reference implementation.
        actual_output = output_xpu.cpu().float()
        clone_output = output_xpu_clone.cpu().float()
        ref_output = torch.empty_like(query)
        self.ref_single_query_cached_kv_attention(
            ref_output,
            query,
            num_queries_per_kv,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            scale,
            alibi_slopes,
        )
        try:
            torch.testing.assert_close(
                actual_output, ref_output.float(), atol=1e-3, rtol=1e-2
            )
            torch.testing.assert_close(
                clone_output, ref_output.float(), atol=1e-3, rtol=1e-2
            )
            print(f"attention {version} accuracy test passed")
        except Exception as e:
            print("accuracy test failed:", e)
            print("expected output: ", ref_output.flatten()[:64])
            print("actual output: ", actual_output.flatten()[:64])

    @pytest.mark.skipif(
        not torch.xpu.has_2d_block_array(),
        reason="Skipped on ATS-M",
    )
    def test(self):
        for version in ["v1", "v2"]:
            self.paged_attention(version)
