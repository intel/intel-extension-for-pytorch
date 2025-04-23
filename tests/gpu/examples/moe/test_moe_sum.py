import torch
import pytest

import intel_extension_for_pytorch  # noqa

dpcpp_device = torch.device("xpu")


class TestTorchMethod:
    def ref_moe_sum(self, input, n_token, n_topk, n_hiddensize):
        return torch.sum(input, dim=1)  # sum alogn n_topk

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("n_token", [2, 32])
    @pytest.mark.parametrize("n_topk", [1, 2])
    @pytest.mark.parametrize("n_hiddensize", [512, 1024])
    def test_moe_sum(self, dtype, n_token, n_topk, n_hiddensize):
        input = torch.randn(
            [n_token, n_topk, n_hiddensize], dtype=dtype, device=dpcpp_device
        )
        output = torch.zeros(
            [input.shape[0], input.shape[2]], dtype=dtype, device=dpcpp_device
        )
        ref_output = self.ref_moe_sum(input, n_token, n_topk, n_hiddensize)
        torch.ops.torch_ipex.moe_sum(input, output)
        # Compare the results
        torch.testing.assert_close(ref_output, output, atol=1e-2, rtol=1e-2)
