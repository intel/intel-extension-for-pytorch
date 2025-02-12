import copy

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa F401
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_embedding_bag_all(self, dtype=torch.float32):
        weight_elem = 56
        for weight_feature_size in [1, 127, 128]:
            for mode in ["sum", "mean", "max"]:
                for include_last_offset in [False, True]:
                    for padding_idx in [None, 29, 10]:
                        embedding = nn.EmbeddingBag(
                            weight_elem,
                            weight_feature_size,
                            mode=mode,
                            scale_grad_by_freq=False,
                            include_last_offset=include_last_offset,
                            padding_idx=padding_idx,
                        )
                        input = torch.Tensor([9, 29, 49, 39, 19, 29, 19, 9, 0]).long()
                        if not include_last_offset:
                            offsets = torch.Tensor([0, 1, 2, 4, 7]).long()
                        else:
                            offsets = torch.Tensor([0, 1, 2, 4, 7, 9]).long()
                        output = embedding(input, offsets)
                        grad_cpu = torch.randn(output.shape, dtype=torch.float)
                        embedding.zero_grad()
                        output.backward(grad_cpu)
                        for param in embedding._parameters.values():
                            grad_weight_cpu = copy.deepcopy(param._grad)
                        input_xpu = input.to("xpu")
                        offsets_xpu = offsets.to("xpu")
                        embedding_xpu = embedding.to("xpu", dtype=dtype)
                        grad_xpu = grad_cpu.to("xpu", dtype=dtype)
                        output_xpu = embedding_xpu(input_xpu, offsets_xpu)
                        embedding_xpu.zero_grad()
                        output_xpu.backward(grad_xpu)
                        for param in embedding_xpu._parameters.values():
                            grad_weight_xpu = copy.deepcopy(param._grad.to("cpu"))
                        print("test cpu", output)
                        print("test xpu", output_xpu.cpu())
                        print("test cpu grad", grad_weight_cpu)
                        print("test xpu grad", grad_weight_xpu.cpu())
                        self.assertEqual(
                            output, output_xpu.cpu().float(), atol=1e-5, rtol=1e-5
                        )
                        # FIXME: Skip max + padding_idx case. No backend implementation.
                        if not (mode == "max" and padding_idx is not None):
                            self.assertEqual(
                                grad_weight_cpu,
                                grad_weight_xpu.cpu().float(),
                                atol=1e-5,
                                rtol=1e-5,
                            )

    def test_embeddingbag_out_of_bounds(self):
        stderr = TestCase.runWithPytorchAPIUsageStderr(
            f"""\
import torch
import intel_extension_for_pytorch   # noqa F401
from torch.testing._internal.common_utils import (run_tests, TestCase)

class TestThatContainsXPUAssert(TestCase):
    def test_embeddingbag_out_of_bounds(self):
        emb = torch.nn.EmbeddingBag(10, 10).to("xpu")
        input = torch.randint(low=20, high=50, size=[1]).to("xpu")
        offset = torch.tensor([0], dtype=torch.int32).to("xpu")
        
        a = emb(input, offset)
        torch.xpu.synchronize()


if __name__ == '__main__':
    run_tests()
        """
        )
        self.assertIn("Assertion `vec_idx < num_row_` failed", stderr)
