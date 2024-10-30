import torch
import intel_extension_for_pytorch as ipex
import random
from itertools import product
from unittest import TestCase
import unittest

NUM_HEADS = [32]
NUM_QUERIES_PER_KV = [1, 4]
HEAD_SIZES = [128, 64]
DTYPES = [torch.bfloat16, torch.float32]


def mha_ref(q, k, v, scale, is_causal):
    if is_causal:
        # bottom corner of the mask
        cur_mask = torch.full(
            (q.size(-2), q.size(-2)), float("-inf"), dtype=q.dtype, device=q.device
        )
        cur_mask = cur_mask.triu(1)
        past_mask = torch.zeros(
            q.size(-2), k.size(-2) - q.size(-2), dtype=q.dtype, device=q.device
        )
        mask = torch.cat([past_mask, cur_mask], dim=-1)
    else:
        mask = None

    kv_groups = q.size(1) // k.size(1)
    k = k.repeat_interleave(kv_groups, dim=1)
    v = v.repeat_interleave(kv_groups, dim=1)
    attn = torch.matmul(q, k.transpose(-2, -1))
    attn = attn * scale
    if is_causal:
        attn = attn + mask
    attn = torch.nn.functional.softmax(attn, dim=-1)
    output1 = torch.matmul(attn, v)
    return output1


class TestFlashAttnVarLen(TestCase):

    @torch.inference_mode()
    def _test_flash_attn_varlen(
        self,
        num_heads: int,
        num_queries_per_kv: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> None:
        random.seed(0)
        torch.manual_seed(0)

        MAX_SEQ_LEN = 1024
        MAX_CTX_LEN = 512
        BS = 8
        num_pages = 1280
        page_size = 16
        max_block_per_request = 128
        query_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(BS)]
        ctx_lens = [random.randint(0, MAX_CTX_LEN) for _ in range(BS)]
        seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
        cu_seq_lens_q = torch.cumsum(
            torch.tensor([0] + query_lens[:], dtype=torch.int32), dim=0
        ).to(torch.int32)
        cu_seq_lens_kv = torch.cumsum(
            torch.tensor([0] + seq_lens[:], dtype=torch.int32), dim=0
        ).to(torch.int32)
        num_kv_heads = num_heads // num_queries_per_kv

        num_tokens = sum(query_lens)
        query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)

        output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

        k_cache = torch.zeros(
            num_pages, num_kv_heads, page_size, head_size, dtype=dtype
        )
        v_cache = torch.zeros(
            num_pages, num_kv_heads, page_size, head_size, dtype=dtype
        )

        kv = torch.randn(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
        keys, values = kv.unbind(dim=1)

        block_ids = torch.arange(0, num_pages, dtype=torch.int32)
        block_ids = block_ids[torch.randperm(num_pages)]
        block_table = block_ids[: BS * max_block_per_request].view(
            BS, max_block_per_request
        )

        # cache the key and value
        for i in range(BS):
            seq_len = seq_lens[i]
            seq_blocks = (seq_len + page_size - 1) // page_size
            key = keys[cu_seq_lens_kv[i] : cu_seq_lens_kv[i + 1]]
            value = values[cu_seq_lens_kv[i] : cu_seq_lens_kv[i + 1]]
            for j in range(seq_blocks):
                start = j * page_size
                end = min(start + page_size, seq_len)
                for k in range(num_kv_heads):
                    k_cache[block_table[i, j], k, : end - start] = key[start:end, k]
                    v_cache[block_table[i, j], k, : end - start] = value[start:end, k]

        max_seq_len_q = max(query_lens)
        max_seq_len_kv = max(seq_lens)
        scale = float(1.0 / (head_size**0.5))

        output_ref = torch.empty_like(query)
        for i in range(BS):
            query_i = query[cu_seq_lens_q[i] : cu_seq_lens_q[i + 1]]
            key_i = keys[cu_seq_lens_kv[i] : cu_seq_lens_kv[i + 1]]
            value_i = values[cu_seq_lens_kv[i] : cu_seq_lens_kv[i + 1]]
            output_i = mha_ref(
                query_i.unsqueeze(0).transpose(1, 2),
                key_i.unsqueeze(0).transpose(1, 2),
                value_i.unsqueeze(0).transpose(1, 2),
                scale,
                is_causal=True,
            )
            output_i = output_i.squeeze(0).transpose(0, 1)
            output_ref[cu_seq_lens_q[i] : cu_seq_lens_q[i + 1]] = output_i

        output = torch.empty_like(query)
        ipex.llm.modules.PagedAttention.flash_attn_varlen_func(
            output,
            query,
            k_cache,
            v_cache,
            cu_seq_lens_q,
            cu_seq_lens_kv,
            max_seq_len_q,
            max_seq_len_kv,
            scale,
            True,
            block_table,
            1.0,
            1.0,
            None,
        )
        assert torch.allclose(
            output_ref, output, atol=1e-6 if dtype == torch.float else 5e-2
        )

    def test_flash_attn_varlen(self):
        for num_heads, num_queries_per_kv, head_size, dtype in product(
            NUM_HEADS, NUM_QUERIES_PER_KV, HEAD_SIZES, DTYPES
        ):
            self._test_flash_attn_varlen(
                num_heads, num_queries_per_kv, head_size, dtype
            )


if __name__ == "__main__":
    torch.manual_seed(2020)
    test = unittest.main()
