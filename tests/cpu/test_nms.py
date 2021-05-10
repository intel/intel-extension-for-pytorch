import unittest, copy
import torch
import torch.nn as nn
import intel_pytorch_extension as ipex
from common_utils import TestCase
import time, sys
from intel_pytorch_extension import batch_score_nms, parallel_scale_back_batch
import torch.nn.functional as F
import os

def get_rand_seed():
    return int(time.time() * 1000000000)

# This function is from https://github.com/kuangliu/pytorch-ssd.
def calc_iou_tensor(box1, box2):
    """ Calculation of IoU based on two boxes tensor,
        Reference to https://github.com/kuangliu/pytorch-ssd
        input:
            box1 (N, 4)
            box2 (M, 4)
        output:
            IoU (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)
    be1 = box1.unsqueeze(1).expand(-1, M, -1)
    be2 = box2.unsqueeze(0).expand(N, -1, -1)
    # Left Top & Right Bottom
    lt = torch.max(be1[:,:,:2], be2[:,:,:2])
    #mask1 = (be1[:,:, 0] < be2[:,:, 0]) ^ (be1[:,:, 1] < be2[:,:, 1])
    #mask1 = ~mask1
    rb = torch.min(be1[:,:,2:], be2[:,:,2:])
    #mask2 = (be1[:,:, 2] < be2[:,:, 2]) ^ (be1[:,:, 3] < be2[:,:, 3])
    #mask2 = ~mask2
    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:,:,0]*delta[:,:,1]
    #*mask1.float()*mask2.float()
    delta1 = be1[:,:,2:] - be1[:,:,:2]
    area1 = delta1[:,:,0]*delta1[:,:,1]
    delta2 = be2[:,:,2:] - be2[:,:,:2]
    area2 = delta2[:,:,0]*delta2[:,:,1]
    iou = intersect/(area1 + area2 - intersect)
    return iou

class TestScaleBackBatch(TestCase):
    def scale_back_batch(self, bboxes_in, scores_in, dboxes_xywh, scale_xy, scale_wh):
        """
            Python implementation of Encoder::scale_back_batch, refer to https://github.com/mlcommons/inference/blob/v0.7/others/cloud/single_stage_detector/pytorch/utils.py
        """
        bboxes_in[:, :, :2] = scale_xy*bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = scale_wh*bboxes_in[:, :, 2:]
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2]*dboxes_xywh[:, :, 2:] + dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp()*dboxes_xywh[:, :, 2:]
        # Transform format to ltrb
        l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                     bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]
        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b
        return bboxes_in, F.softmax(scores_in, dim=-1)

    def test_scale_back_batch_result(self):
        batch_size = 16
        number_boxes = 1024
        scale_xy = 0.1
        scale_wh = 0.2
        predicted_loc = torch.randn((batch_size, number_boxes, 4)).contiguous().to(torch.float32)
        predicted_score = torch.randn((batch_size, number_boxes, 81)).contiguous().to(torch.float32)
        dboxes_xywh = torch.randn((1, number_boxes, 4)).contiguous().to(torch.float64)
        bbox_res1, score_res1 = self.scale_back_batch(predicted_loc.clone(), predicted_score.clone(), dboxes_xywh.clone(), scale_xy, scale_wh)
        bbox_res2, score_res2 = parallel_scale_back_batch(predicted_loc, predicted_score, dboxes_xywh, scale_xy, scale_wh)
        self.assertTrue(torch.allclose(bbox_res1, bbox_res2, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(score_res1, score_res2, rtol=1e-4, atol=1e-4))

class TestNMS(TestCase):
    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        """
            Python implementation of Encoder::decode_single, refer to https://github.com/mlcommons/inference/blob/v0.7/others/cloud/single_stage_detector/pytorch/utils.py
        """
        # perform non-maximum suppression
        # Reference to https://github.com/amdegroot/ssd.pytorch

        bboxes_out = []
        scores_out = []
        labels_out = []
        for i, score in enumerate(scores_in.split(1, 1)):
            # skip background
            # print(score[score>0.90])
            if i == 0: continue
            score = score.squeeze(1)
            mask = score > 0.05
            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0: continue
            score_sorted, score_idx_sorted = score.sort(dim=0)
            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []
            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()
                # we only need iou < criteria
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i]*len(candidates))
        bboxes_out, labels_out, scores_out = torch.cat(bboxes_out, dim=0), \
               torch.tensor(labels_out, dtype=torch.long), \
               torch.cat(scores_out, dim=0)
        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]

    def test_nms_result(self):
        batch_size = 1
        number_boxes = 15130
        scale_xy = 0.1
        scale_wh = 0.2
        criteria = 0.50
        max_output = 200
        predicted_loc = torch.randn((batch_size, number_boxes, 4)).contiguous().to(torch.float32)
        predicted_score = torch.randn((batch_size, number_boxes, 81)).contiguous().to(torch.float32)
        dboxes_xywh = torch.randn((1, number_boxes, 4)).contiguous().to(torch.float64)
        dboxes_xywh = torch.load(os.path.dirname(__file__) + "/data/nms_dboxes_xywh.pt")
        bboxes, probs = parallel_scale_back_batch(predicted_loc, predicted_score, dboxes_xywh, scale_xy, scale_wh)
        bboxes_clone = bboxes.clone()
        probs_clone = probs.clone()

        output = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(self.decode_single(bbox, prob, criteria, max_output))
        output2 = batch_score_nms(bboxes_clone, probs_clone, criteria, max_output)

        for i in range(batch_size):
            loc, label, prob = [r for r in output[i]]
            loc2, label2, prob2 = [r for r in output2[i]]
            self.assertTrue(torch.allclose(loc, loc2, rtol=1e-4, atol=1e-4))
            self.assertEqual(label, label2)
            self.assertTrue(torch.allclose(prob, prob2, rtol=1e-4, atol=1e-4))

if __name__ == '__main__':
    test = unittest.main()
