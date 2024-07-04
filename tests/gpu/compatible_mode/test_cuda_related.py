import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch as ipex  # noqa
ipex.compatible_mode()

cuda_device = torch.device("cuda")


class TestTorchMethod(TestCase):
    def test_is_nccl_available(self):
        x = torch.distributed.is_nccl_available()
        self.assertEqual(x, True)

    def test_is_bf16_supported(self):
        x = torch.cuda.is_bf16_supported()
        self.assertEqual(x, True)
