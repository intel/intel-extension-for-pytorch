import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch as ipex  # noqa
ipex.compatible_mode()

cuda_device = torch.device("cuda")


class TestTorchMethod(TestCase):
    def test_is_initialized(self):
        if torch.cuda.is_available():
            torch.cuda.init()
            x = torch.cuda.is_initialized()
            self.assertEqual(x, True)

    def test_is_in_bad_fork(self):
        x = torch.cuda._is_in_bad_fork()
        self.assertEqual(x, False)

    def test_lazy_init(self):
        torch.cuda._lazy_init()
