import torch
import pytest

import intel_extension_for_pytorch  # noqa

dpcpp_device = torch.device("xpu")


class TestTorchMethod:
    def ref_topk_softmax(self, gating_logits, n_topk):
        gating_logits = gating_logits.to(torch.float)
        softmax = torch.nn.functional.softmax(gating_logits, dim=-1, dtype=torch.float)
        topk_weights, topk_indices = torch.topk(softmax, n_topk, dim=-1)

        # token_for_expert: # of tokens for each expert
        # token_offset: the offset of each token for each export
        n_experts = gating_logits.shape[-1]
        token_for_experts = torch.zeros(
            n_experts, device=dpcpp_device, dtype=torch.int32
        )
        for i in range(n_experts):
            token_for_experts[i] = (topk_indices == i).sum().item()

        return topk_weights, topk_indices, token_for_experts

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("n_token", [2, 32, 4096])
    @pytest.mark.parametrize("n_expert", [8, 32])
    @pytest.mark.parametrize("n_topk", [1, 2, 4, 8])
    def test_topk_softmax(self, dtype, n_token, n_topk, n_expert):
        gating_logits = torch.randn(n_token, n_expert, device=dpcpp_device)

        ref_token_weights, ref_topk_indices, ref_token_for_experts = (
            self.ref_topk_softmax(gating_logits, n_topk)
        )
        topk_weights, topk_indices, token_for_experts, _ = (
            torch.ops.torch_ipex.topk_softmax(gating_logits, n_topk)
        )

        # Compare the results
        torch.testing.assert_close(
            ref_token_weights, topk_weights, atol=1e-2, rtol=1e-2
        )
        assert torch.equal(ref_topk_indices, topk_indices)
        assert torch.equal(ref_token_for_experts, token_for_experts)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("full_nan", [True, False])
    def test_topk_softmax_nan(
        self,
        dtype,
        full_nan,
    ):
        n_token = 32
        n_expert = 32
        n_topk = 8

        if full_nan:
            gating_logits = torch.full(
                (n_token, n_expert), float("nan"), device=dpcpp_device, dtype=dtype
            )
        else:
            gating_logits = torch.randn(
                n_token, n_expert, device=dpcpp_device, dtype=dtype
            )
            gating_logits[0][0] = float("nan")

        topk_weights, topk_indices, token_for_experts, expert_offsets = (
            torch.ops.torch_ipex.topk_softmax(gating_logits, n_topk)
        )

        assert torch.all(topk_indices < n_expert)
        assert torch.all(token_for_experts <= n_token * n_topk)
        assert torch.all(expert_offsets < n_token * n_topk)
