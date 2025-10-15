import torch
import pytest
import random

import intel_extension_for_pytorch  # noqa


class TestTorchMethod:

    def random_pair(self, N):
        A = random.randint(0, N - 1)
        B = random.randint(1, N - A)
        return A, B

    def moe_rows_counts(
        self, topk_indices, n_tokens, experts_offset, n_experts_local, n_topk
    ):
        topk_indices = topk_indices.view(-1)
        n_experts_aligned = (n_experts_local + 7) // 8 * 8
        rows_for_experts = torch.zeros(n_experts_aligned, dtype=torch.int32)
        offsets = torch.full_like(topk_indices, -1)

        for global_id in range(n_tokens * n_topk):
            token_idx = global_id // n_topk
            topk_idx = global_id % n_topk
            expert_id = topk_indices[global_id] - experts_offset

            if expert_id < 0 or expert_id >= n_experts_local:
                offsets[global_id] = -1
                continue

            old = rows_for_experts[expert_id].item()
            rows_for_experts[expert_id] += 1
            offsets[global_id] = old

        return rows_for_experts, offsets.view(n_tokens, n_topk)

    @pytest.mark.parametrize("tokens", [1, 32, 1024])
    @pytest.mark.parametrize("topk", [4, 8])
    @pytest.mark.parametrize("n_experts", [8, 32, 128])
    def test_moe_rows_counts(self, tokens, topk, n_experts):
        torch.manual_seed(0)
        selected_experts = torch.stack(
            [torch.randperm(n_experts, device="xpu")[:topk] for _ in range(tokens)]
        ).to(torch.int32)

        experts_offset, n_expert_local = self.random_pair(n_experts)

        rows_for_experts, expert_offsets = torch.ops.torch_ipex.moe_rows_counts(
            selected_experts, experts_offset, n_expert_local
        )

        ref_rows_for_experts, ref_expert_offsets = self.moe_rows_counts(
            selected_experts, tokens, experts_offset, n_expert_local, topk
        )

        assert torch.equal(ref_rows_for_experts, rows_for_experts.to("cpu"))
