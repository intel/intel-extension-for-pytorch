import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


class TestNNMethod(TestCase):
    def test_nms(self):
        box = torch.FloatTensor([[2, 3.1, 1, 7], [3, 4, 8, 4.8], [4, 4, 5.6, 7],
                                 [0.1, 0, 8, 1], [4, 4, 5.7, 7.2]]).xpu()
        score = torch.FloatTensor([0.5, 0.3, 0.2, 0.4, 0.3]).xpu()
        out_ref = torch.LongTensor([0, 3, 1, 4])
        out = torch.xpu.nms(box, score, 0.3)
        self.assertEqual(out.cpu(), out_ref)
