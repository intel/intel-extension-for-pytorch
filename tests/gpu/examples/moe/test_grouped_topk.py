import torch
import pytest
from typing import Optional
import intel_extension_for_pytorch  # noqa

dpcpp_device = torch.device("xpu")


class TestTorchMethod:
    def ref_grouped_topk(
        self,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        num_expert_group: int = 0,
        topk_group: int = 0,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if scoring_func == "softmax":
            gating_output = gating_output.to(torch.float32)
            scores = torch.softmax(gating_output, dim=-1)
        elif scoring_func == "sigmoid":
            scores = gating_output.sigmoid()
        else:
            raise ValueError(f"Unsupported scoring function: {scoring_func}")

        num_token = scores.shape[0]
        if e_score_correction_bias is not None:
            # Store original scores before applying correction bias. We use biased
            # scores for expert selection but original scores for routing weights
            e_score_correction_bias = e_score_correction_bias.to(torch.float32)
            original_scores = scores
            scores = scores + e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores.view(num_token, num_expert_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
        else:
            group_scores = (
                scores.view(num_token, num_expert_group, -1).max(dim=-1).values
            )  # [n, n_group]
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=True)[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
            .reshape(num_token, -1)
        )  # [n, e]
        tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

        if e_score_correction_bias is not None:
            topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=True)[1]
            # Use original unbiased scores for the routing weights
            topk_weights = original_scores.gather(1, topk_ids)
        else:
            topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=True)

        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    @pytest.mark.parametrize("seed", [123, 356, 478])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n_token", [1, 2, 64, 256])
    @pytest.mark.parametrize("n_expert", [64, 128, 256])
    @pytest.mark.parametrize("n_topk", [4, 6, 8])
    @pytest.mark.parametrize("n_topk_group", [4, 6, 8])
    @pytest.mark.parametrize("n_expert_group", [8])
    @pytest.mark.parametrize("renormalize", [True, False])
    @pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
    @pytest.mark.parametrize("has_bias", [True, False])
    def test_grouped_topk(
        self,
        seed,
        dtype,
        n_token,
        n_expert,
        n_topk,
        n_expert_group,
        n_topk_group,
        renormalize,
        scoring_func,
        has_bias,
    ):

        torch.manual_seed(seed)
        gating_output = torch.randn(n_token, n_expert, device=dpcpp_device).to(dtype)
        hidden_states = torch.zeros(n_token, n_expert, device=dpcpp_device).to(dtype)
        bias = None
        if has_bias:
            if has_bias and scoring_func == "sigmoid" and dtype is not torch.float32:
                # Low-precision sigmoid calculation errors can cause the results to fluctuate too much
                # using a bias of bigger number to avoid this
                bias = torch.arange(1, n_expert + 1).to(dpcpp_device).to(dtype)
            else:
                bias = torch.randn(n_expert, device=dpcpp_device).to(dtype)

        ref_topk_weights, ref_topk_indices = self.ref_grouped_topk(
            gating_output,
            n_topk,
            renormalize,
            n_expert_group,
            n_topk_group,
            scoring_func,
            bias,
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.XPU,
            ],
            record_shapes=True,
        ) as prof:
            topk_weights, topk_indices, _, _ = torch.ops.torch_ipex.grouped_topk(
                hidden_states,
                gating_output,
                n_topk,
                renormalize,
                n_expert_group,
                n_topk_group,
                scoring_func,
                bias,
            )

        # Compare the results
        torch.testing.assert_close(ref_topk_weights, topk_weights, atol=2e-2, rtol=1e-2)
        assert torch.equal(ref_topk_indices, topk_indices)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("renormalize", [True])
    @pytest.mark.parametrize("full_nan", [True, False])
    def test_grouped_topk_sigmoid_nan(
        self,
        dtype,
        renormalize,
        full_nan,
    ):
        n_token = 512
        n_expert = 256
        n_topk = 8
        n_expert_group = 8
        n_topk_group = 4

        gating_output = torch.randn(n_token, n_expert, device=dpcpp_device).to(dtype)
        hidden_states = torch.zeros(n_token, n_expert, device=dpcpp_device).to(dtype)
        bias = torch.randn(n_expert, device=dpcpp_device).to(dtype)

        if full_nan:
            gating_output = torch.full(
                gating_output.size(), float("nan"), device=dpcpp_device, dtype=dtype
            ).contiguous()
        else:
            gating_output[0][0] = float("nan")

        topk_weights, topk_indices, token_for_experts, expert_offsets = (
            torch.ops.torch_ipex.grouped_topk(
                hidden_states,
                gating_output,
                n_topk,
                renormalize,
                n_expert_group,
                n_topk_group,
                "sigmoid",
                bias,
            )
        )

        assert torch.all(topk_indices < n_expert)
        assert torch.all(token_for_experts <= n_token * n_topk)
        assert torch.all(expert_offsets < n_token * n_topk)
