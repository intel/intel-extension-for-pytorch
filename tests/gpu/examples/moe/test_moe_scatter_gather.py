import torch
import pytest
import random

import intel_extension_for_pytorch  # noqa

dpcpp_device = torch.device("xpu")


class TestTorchMethod:

    def init(self, n_token, n_expert, n_topk, experts_offset, n_expert_local):
        gating_logits = torch.rand(n_token, n_expert, device=dpcpp_device)
        gating_logits = gating_logits.to(torch.float)
        softmax = torch.nn.functional.softmax(gating_logits, dim=-1, dtype=torch.float)
        topk_weights, topk_indices = torch.topk(softmax, n_topk, dim=-1)
        topk_indices = topk_indices.to(torch.int32)
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

        # token_for_expert: # of tokens for each expert
        # token_offset: the offset of each token for each export
        n_experts = gating_logits.shape[-1]
        token_for_experts = torch.zeros(
            n_experts, device=dpcpp_device, dtype=torch.int32
        )
        token_offset = torch.empty_like(topk_indices, dtype=torch.int32)
        token_offset.fill_(-1)
        for i in range(n_expert_local):
            token_for_experts[i] = (topk_indices == (i + experts_offset)).sum().item()
            mask = topk_indices == (i + experts_offset)
            exclusive_mask = (
                torch.cumsum(mask.flatten(), dim=0, dtype=torch.int32).reshape(
                    mask.shape
                )
                - mask.to(torch.int32)
                + 1
            )
            token_offset[mask] += exclusive_mask[mask]

        return topk_weights, topk_indices, token_for_experts, token_offset

    def random_pair(self, N):
        A = random.randint(0, N - 1)
        B = random.randint(0, N - A)
        return A, B

    def ref_moe_scatter(
        self,
        topk_indices,
        token_for_experts,
        token_offset,
        experts_offset,
        n_expert_local,
    ):
        # inclusive scan
        token_for_experts = torch.cumsum(token_for_experts, dim=0, dtype=torch.int32)

        n_token = topk_indices.shape[0]
        n_topk = topk_indices.shape[-1]
        mapped_slot = torch.zeros(
            (n_token, n_topk), dtype=torch.int32, device=topk_indices.device
        )
        for i in range(n_token):
            for j in range(n_topk):
                expert_id = topk_indices[i, j]
                if (
                    expert_id < experts_offset
                    or expert_id >= experts_offset + n_expert_local
                ):
                    mapped_slot[i, j] = -1
                    continue
                expert_id -= experts_offset
                expert_offset = token_for_experts[expert_id - 1] if expert_id > 0 else 0
                slot_id = token_offset[i, j] + expert_offset
                mapped_slot[i, j] = slot_id
        return mapped_slot

    def ref_moe_gather(
        self,
        activation,
        topk_weights,
        mapped_slot,
        token_for_experts,
        n_expert,
        n_topk,
        n_channels,
        normalize_scale,
    ):
        # normalize topk_weights along the last dimension
        n_token = topk_weights.shape[0]
        if normalize_scale:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        gathered = torch.zeros(
            (n_token, n_channels), dtype=activation.dtype, device=activation.device
        )
        for i in range(n_token):
            slot_id = mapped_slot[i, 0]
            if slot_id != -1:
                gathered[i] = topk_weights[i, 0] * activation[slot_id]

            for j in range(1, n_topk):
                slot_id = mapped_slot[i, j]
                if slot_id == -1:
                    continue
                gathered[i] += topk_weights[i, j] * activation[slot_id]

        return gathered

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("n_expert", [8, 16])
    @pytest.mark.parametrize("n_token", [16, 32, 4096])
    @pytest.mark.parametrize("n_topk", [1, 2, 4, 6, 8])
    def test_moe_scatter(self, dtype, n_expert, n_token, n_topk):
        n_channels = 1024
        experts_offset, n_expert_local = self.random_pair(n_expert)

        _, topk_indices, token_for_experts, token_offset = self.init(
            n_token, n_expert, n_topk, experts_offset, n_expert_local
        )
        topk_indices_ref = topk_indices.clone()

        ref_mapped_slot = self.ref_moe_scatter(
            topk_indices_ref,
            token_for_experts,
            token_offset,
            experts_offset,
            n_expert_local,
        )

        activation = torch.randn(
            (n_token, n_channels), dtype=dtype, device=dpcpp_device
        )

        _, mapped_slot = torch.ops.torch_ipex.moe_scatter(
            activation,
            token_for_experts,
            topk_indices,
            token_offset,
            experts_offset,
            n_expert_local,
            n_topk,
        )

        # Compare the results
        assert torch.equal(ref_mapped_slot, mapped_slot)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("n_expert", [8, 16])
    @pytest.mark.parametrize("n_token", [16, 32, 4096])
    @pytest.mark.parametrize("n_topk", [1, 2, 4, 6, 8])
    def test_moe_gather(self, dtype, n_expert, n_token, n_topk):
        n_channels = 1024
        experts_offset, n_expert_local = self.random_pair(n_expert)
        activation = torch.rand(
            (n_token * n_topk, n_channels), dtype=dtype, device=dpcpp_device
        )
        topk_weights, topk_indices, token_for_experts, token_offset = self.init(
            n_token, n_expert, n_topk, experts_offset, n_expert_local
        )
        mapped_slot = self.ref_moe_scatter(
            topk_indices,
            token_for_experts,
            token_offset,
            experts_offset,
            n_expert_local,
        )

        normalize_scale = True
        ref_gathered = self.ref_moe_gather(
            activation,
            topk_weights,
            mapped_slot,
            token_for_experts,
            n_expert,
            n_topk,
            n_channels,
            normalize_scale,
        )
        gathered = torch.ops.torch_ipex.moe_gather(
            activation,
            topk_weights,
            mapped_slot,
            token_for_experts,
            n_expert,
            n_topk,
            normalize_scale,
        )

        # Compare the results
        torch.testing.assert_close(ref_gathered, gathered, rtol=1e-2, atol=1e-2)
