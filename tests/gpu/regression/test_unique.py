import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa


class TestTorchMethod(TestCase):
    def test_unique_non_contiguous(self):
        det = torch.tensor([[3.7, 3.1, 5.6, 3.6, 8.9, 0.0], [5.4, 1.0, 5.5, 3.5, 8.0, 0.0], [2.1, 2.2, 2.6, 3.7, 7.4, 2.7], [
                           5.5, 2.5, 6.3, 3.6, 3.5, 0.0], [4.9, 1.6, 6.1, 2.1, 3.1, 2.7]], device='cpu')
        det_xpu = det.to("xpu")
        output = det[:, -1].unique()
        output_xpu = det_xpu[:, -1].unique()
        self.assertEqual(output, output_xpu.cpu())
