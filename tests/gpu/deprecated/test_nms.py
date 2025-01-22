import torch
from torch.testing._internal.common_utils import TestCase
import torchvision

import intel_extension_for_pytorch  # noqa


class TestNNMethod(TestCase):
    def test_nms(self):
        for dtype in [torch.float, torch.half, torch.bfloat16]:
            box = (
                torch.FloatTensor(
                    [
                        [2, 3.1, 1, 7],
                        [3, 4, 8, 4.8],
                        [4, 4, 5.6, 7],
                        [0.1, 0, 8, 1],
                        [4, 4, 5.7, 7.2],
                    ]
                )
                .xpu()
                .to(dtype)
            )
            score = torch.FloatTensor([0.5, 0.3, 0.2, 0.4, 0.3]).xpu().to(dtype)
            out_ref = torch.LongTensor([0, 3, 1, 4])
            out = torchvision.ops.nms(box, score, 0.3)
            self.assertEqual(out.cpu(), out_ref)

    def test_batched_nms(self):
        box1 = torch.FloatTensor(
            [
                [2, 3.1, 1, 7],
                [3, 4, 8, 4.8],
                [4, 4, 5.6, 7],
                [0.1, 0, 8, 1],
                [4, 4, 5.7, 7.2],
            ]
        )
        score1 = torch.FloatTensor([0.5, 0.3, 0.2, 0.4, 0.3])
        idx1 = torch.LongTensor([2, 1, 3, 4, 0])
        box2 = torch.FloatTensor(
            [
                [2, 3.1, 1, 5],
                [3, 4, 8, 4.8],
                [4, 4, 5.6, 7],
                [0.1, 0, 6, 1],
                [4, 4, 5.7, 7.2],
            ]
        )
        score2 = torch.FloatTensor([0.5, 0.1, 0.2, 0.4, 0.8])
        idx2 = torch.LongTensor([0, 1, 2, 4, 3])
        boxes = torch.cat([box1, box2], dim=0).xpu()
        scores = torch.cat([score1, score2], dim=0).xpu()
        idxs = torch.cat([idx1, idx2], dim=0).xpu()
        out = torchvision.ops.batched_nms(boxes, scores, idxs, 0.3)
        out_ref = torch.LongTensor([9, 0, 5, 3, 1, 4, 7])
        self.assertEqual(out.cpu(), out_ref)
