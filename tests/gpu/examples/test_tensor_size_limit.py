import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest



class TestNNMethod(TestCase):
    @pytest.mark.skipif(
        torch.xpu.utils.has_2d_block_array(),
        reason="Tensor size limit checking is only required for ATSM"
    )
    def test_tensor_size_limit(self):
        with self.assertRaisesRegex(RuntimeError, 'Current platform can NOT allocate memory block with size larger than 4GB!.*'):
            large_tensor = torch.randn([1, 4294967296 + 1], dtype=torch.int8, device="xpu")  # 1 byte larger than 4GB
