import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

class TestTorchMethod(TestCase):
    def test_itt(self, dtype=torch.float):

        #
        # Test itt func.
        #
        print("Testing itt func.!\n")
        output = torch.rand(4, 3, 640, 1024)
        output = output.to('xpu')
        with torch.xpu.emit_itt():
            torch.max(output, dim=1)
