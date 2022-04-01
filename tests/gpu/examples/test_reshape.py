import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch
import pytest


class TestTorchMethod(TestCase):

    @pytest.mark.skipif("not torch.xpu.using_layout_opt()", reason="only for block format")
    def test_reshape_cat(self, dtype=torch.float):
        """
        Issue: When run SSD-ResNet50 with block format, it raised a shape mismatch error in cat op.
        Root cause: There was no to_plain_if_needed() in reshape_alias. Without it, the size of tensor was changed by reshape but dims was kept unchanged. It led to the shape mismatch error in the next op cat.
        Fixed PR: https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu/pull/709
        """
        input1 = torch.randn([32, 3, 300, 300]).to("xpu")
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False).to("xpu")
        conv_out1 = conv1(input1)
        reshape_out1 = conv_out1.reshape(1, 3, 300, 300, 32)

        input2 = torch.randn([8, 3, 300, 300]).to("xpu")
        conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False).to("xpu")
        conv_out2 = conv2(input2)
        reshape_out2 = conv_out2.reshape(1, 3, 300, 300, 8)

        cat_out = torch.cat((reshape_out1, reshape_out2), -1)
        # Force wait for previous concat to avoid random segfault
        # If exit the case before the kernel is finished
        torch.xpu.synchronize()

        self.assertEqual(cat_out.shape, torch.Size([1, 3, 300, 300, 40]))
