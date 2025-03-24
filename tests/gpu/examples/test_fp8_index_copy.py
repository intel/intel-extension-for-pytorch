################################################################################
# Copyright (C) 2025 Intel Corporation
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written
# permission. This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly
# stated in the License.
################################################################################
import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch as ipex  # noqa
from intel_extension_for_pytorch.quantization.fp8.util import (
    cast_to_fp8,
)


class TestTorchMethod(TestCase):
    def test_index_copy(self, dtype=torch.float):
        a = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5],
            ],
            dtype=dtype,
        ).xpu()
        index = torch.LongTensor([0, 2, 1, 4, 3]).xpu()

        fp8_dtype = ipex._C.Float8Format.kFloat8_E4M3
        b = torch.zeros((5, 5), dtype=dtype).xpu()

        a_scale, a_amax, a_scale_inv = (
            torch.ones([]).xpu(),
            torch.zeros([]).xpu(),
            torch.ones([]).xpu(),
        )

        fp8_meta = {}
        fp8_meta["test"] = ipex._C.FP8TensorMeta()
        fp8_meta["test"].scale = (torch.ones(1) * 2).xpu()
        fp8_meta["test"].scale_inv = torch.ones(1).xpu()
        fp8_meta["test"].amax_history = torch.ones([1, 4]).xpu()

        fp8_tensor = ipex._C.FP8FwdTensors.GEMM1_INPUT

        # print("a = ", a)
        # print("index = ", index)
        a_quantized = cast_to_fp8(a, fp8_meta["test"], fp8_tensor, fp8_dtype)
        b = cast_to_fp8(b, fp8_meta["test"], fp8_tensor, fp8_dtype)
        # print("a_quantized = ", a_quantized)

        # b_cpu = b.cpu()
        # print("b_cpu = ", b_cpu)
        # b_cpu.index_copy_(0, index.cpu(), a_quantized.cpu())
        b_ref = a.clone()
        b_ref.index_copy_(0, index, a)
        b_ref = cast_to_fp8(b_ref, fp8_meta["test"], fp8_tensor, fp8_dtype)
        # print("b_ref = ", b_ref)

        b.index_copy_(0, index, a_quantized)
        # print("b_xpu = ", b)

        self.assertEqual(b.cpu(), b_ref.cpu())
