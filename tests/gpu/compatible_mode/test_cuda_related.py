import torch
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

    def test_tensor_is_cuda(self):
        x = torch.rand(2, 3, device="cuda")
        self.assertEqual(x.is_cuda(), True)

    def test_cuda_non_blocking(self):
        a = torch.rand(1, 2, 3, 4)
        b = a.cuda(non_blocking=True)
        self.assertEqual(b.is_cuda(), True)
