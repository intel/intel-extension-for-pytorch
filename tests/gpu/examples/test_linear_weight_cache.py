import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import ipex
import pytest
import numpy as np

class TestNNMethod(TestCase):
    def test_linear_weight_cache(self):
        '''
        Regression desc:
        When to_plain is used in as_strided under inference mode, RuntimeError
        `Cannot set version_counter for inference tensor` complains.
        The original weight tesnor has version_counter when it is created (for
        its semantics).
        The op to_plain copies the metadata(version_counter) to cache, but the
        created cache is an inf tensor under the context of inf mode.
        Version_counter should not be in an inf tensor, this results in RuntimeError.
        '''
        linear = nn.Linear(13, 512)
        example_input = torch.randn([32768, 13], device="xpu")
        linear.to("xpu")
        with torch.inference_mode():
            for i in range(2):
                res_xpu = linear(example_input)
        example_input = example_input.to("cpu")
        linear.to("cpu")
        with torch.inference_mode():
            for i in range(2):
                res_cpu = linear(example_input)
        self.assertEqual(res_xpu.to("cpu"), res_cpu)
