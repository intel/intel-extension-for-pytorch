import torch
import pytest

import intel_extension_for_pytorch  # noqa

dpcpp_device = torch.device("xpu")


class TestTorchMethod:
    def ref_grouped_topk_sigmoid(
        self,
        gating_output,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        e_score_correction_bias,
        scoring_func="sigmoid",
    ):
        gating_output = gating_output.to(torch.float32)
        if scoring_func == "softmax":
            scores = torch.softmax(gating_output, dim=-1)
        elif scoring_func == "sigmoid":
            scores = gating_output.sigmoid()
        else:
            raise ValueError(f"Unsupported scoring function: {scoring_func}")

        if e_score_correction_bias is not None:
            scores.add_(e_score_correction_bias.unsqueeze(0))

        score_copy = scores.clone()
        num_token = scores.shape[0]
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )  # [n, n_group]
        max_copy = group_scores.clone()
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
            1
        ]  # [n, top_k_group]

        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
            .reshape(num_token, -1)
        )  # [n, e]
        tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # token_for_expert: # of tokens for each expert
        # token_offset: the offset of each token for each export
        n_experts = gating_output.shape[-1]
        token_for_experts = torch.zeros(
            n_experts, device=dpcpp_device, dtype=torch.int32
        )
        for i in range(n_experts):
            token_for_experts[i] = (topk_ids == i).sum().item()

        return (
            topk_weights.to(torch.float32),
            topk_ids.to(torch.int32),
            token_for_experts,
        )

    @pytest.mark.parametrize("seed", [123, 356, 478])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n_token", [1, 2, 64, 256])
    @pytest.mark.parametrize("n_expert", [64, 128, 256])
    @pytest.mark.parametrize("n_topk", [4, 6, 8])
    @pytest.mark.parametrize("n_topk_group", [4, 6, 8])
    @pytest.mark.parametrize("n_expert_group", [8])
    @pytest.mark.parametrize("renormalize", [True])
    @pytest.mark.parametrize("sorted", [True, False])
    def test_grouped_topk_sigmoid(
        self,
        seed,
        dtype,
        n_token,
        n_expert,
        n_topk,
        n_expert_group,
        n_topk_group,
        renormalize,
        sorted,
    ):

        torch.manual_seed(seed)
        gating_output = torch.randn(n_token, n_expert, device=dpcpp_device).to(dtype)
        hidden_states = torch.zeros(n_token, n_expert, device=dpcpp_device).to(dtype)
        bias = torch.randn(n_expert, device=dpcpp_device).to(dtype)

        ref_topk_weights, ref_topk_indices, ref_token_for_experts = (
            self.ref_grouped_topk_sigmoid(
                gating_output, n_topk, renormalize, n_expert_group, n_topk_group, bias
            )
        )
        topk_weights, topk_indices, token_for_experts, _ = (
            torch.ops.torch_ipex.grouped_topk_sigmoid(
                hidden_states,
                gating_output,
                n_topk,
                renormalize,
                n_expert_group,
                n_topk_group,
                "sigmoid",
                bias,
                sorted,
            )
        )
        ref_topk_weights_sorted, _ = torch.sort(ref_topk_weights, -1, descending=True)
        topk_weights_sorted, _ = (
            (topk_weights, None)
            if sorted
            else torch.sort(topk_weights, -1, descending=True)
        )
        ref_topk_indices_sorted, _ = torch.sort(ref_topk_indices, -1, descending=True)
        topk_indices_sorted, _ = torch.sort(topk_indices, -1, descending=True)
        # Compare the results
        torch.testing.assert_close(
            ref_topk_weights_sorted, topk_weights_sorted, atol=2e-2, rtol=1e-2
        )
        # assert torch.equal(ref_topk_indices_sorted, topk_indices_sorted)
        # assert torch.equal(ref_token_for_experts, token_for_experts)
