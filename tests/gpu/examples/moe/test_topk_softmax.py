import torch
import pytest

import intel_extension_for_pytorch  # noqa

dpcpp_device = torch.device("xpu")


class TestTorchMethod:
    def ref_topk_softmax(self, gating_logits, n_topk):
        gating_logits = gating_logits.to(torch.float)
        softmax = torch.nn.functional.softmax(gating_logits, dim=-1, dtype=torch.float)
        topk_weights, topk_indices = torch.topk(softmax, n_topk, dim=-1)

        return topk_weights, topk_indices

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("n_token", [2, 32, 4096])
    @pytest.mark.parametrize("n_expert", [8, 32])
    @pytest.mark.parametrize("n_topk", [1, 2, 4, 8])
    def test_topk_softmax(self, dtype, n_token, n_topk, n_expert):
        gating_logits = torch.randn(n_token, n_expert, device=dpcpp_device)

        ref_token_weights, ref_topk_indices = self.ref_topk_softmax(
            gating_logits, n_topk
        )
        topk_weights, topk_indices = torch.ops.torch_ipex.topk_softmax(
            gating_logits, n_topk, False
        )

        # Compare the results
        torch.testing.assert_close(
            ref_token_weights, topk_weights, atol=1e-2, rtol=1e-2
        )
        assert torch.equal(ref_topk_indices, topk_indices)

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

        topk_weights, topk_indices = torch.ops.torch_ipex.topk_softmax(
            gating_logits, n_topk, False
        )

        assert torch.all(topk_indices < n_expert)
        assert torch.all(topk_indices >= 0)
