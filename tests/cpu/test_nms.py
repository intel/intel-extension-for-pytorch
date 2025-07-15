import unittest
import torch
import torch.nn as nn
from common_utils import TestCase
import time
import torch.nn.functional as F
import os


def nms(dets, scores, threshold, sorted=False):
    return torch.ops.torch_ipex.nms(dets, scores, threshold, sorted)


batch_score_nms = torch.ops.torch_ipex.batch_score_nms
parallel_scale_back_batch = torch.ops.torch_ipex.parallel_scale_back_batch
rpn_nms = torch.ops.torch_ipex.rpn_nms


def get_rand_seed():
    return int(time.time() * 1000000000)


# This function is from https://github.com/kuangliu/pytorch-ssd.
def calc_iou_tensor(box1, box2):
    """Calculation of IoU based on two boxes tensor,
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
    lt = torch.max(be1[:, :, :2], be2[:, :, :2])
    # mask1 = (be1[:,:, 0] < be2[:,:, 0]) ^ (be1[:,:, 1] < be2[:,:, 1])
    # mask1 = ~mask1
    rb = torch.min(be1[:, :, 2:], be2[:, :, 2:])
    # mask2 = (be1[:,:, 2] < be2[:,:, 2]) ^ (be1[:,:, 3] < be2[:,:, 3])
    # mask2 = ~mask2
    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:, :, 0] * delta[:, :, 1]
    # *mask1.float()*mask2.float()
    delta1 = be1[:, :, 2:] - be1[:, :, :2]
    area1 = delta1[:, :, 0] * delta1[:, :, 1]
    delta2 = be2[:, :, 2:] - be2[:, :, :2]
    area2 = delta2[:, :, 0] * delta2[:, :, 1]
    iou = intersect / (area1 + area2 - intersect)
    return iou


class TestScaleBackBatch(TestCase):
    def scale_back_batch(self, bboxes_in, scores_in, dboxes_xywh, scale_xy, scale_wh):
        """
        Python implementation of Encoder::scale_back_batch, refer to \
            https://github.com/mlcommons/inference/blob/v0.7/others/cloud/single_stage_detector/pytorch/utils.py
        """
        bboxes_in[:, :, :2] = scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = scale_wh * bboxes_in[:, :, 2:]
        bboxes_in[:, :, :2] = (
            bboxes_in[:, :, :2] * dboxes_xywh[:, :, 2:] + dboxes_xywh[:, :, :2]
        )
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * dboxes_xywh[:, :, 2:]
        # Transform format to ltrb
        l, t, r, b = (
            bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2],
            bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3],
            bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2],
            bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3],
        )
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
        predicted_loc = (
            torch.randn((batch_size, number_boxes, 4)).contiguous().to(torch.float32)
        )
        predicted_score = (
            torch.randn((batch_size, number_boxes, 81)).contiguous().to(torch.float32)
        )
        dboxes_xywh = torch.randn((1, number_boxes, 4)).contiguous().to(torch.float64)
        bbox_res1, score_res1 = self.scale_back_batch(
            predicted_loc.clone(),
            predicted_score.clone(),
            dboxes_xywh.clone(),
            scale_xy,
            scale_wh,
        )
        bbox_res2, score_res2 = parallel_scale_back_batch(
            predicted_loc, predicted_score, dboxes_xywh, scale_xy, scale_wh
        )
        # test autocast
        with torch.cpu.amp.autocast():
            bbox_res3, score_res3 = parallel_scale_back_batch(
                predicted_loc, predicted_score, dboxes_xywh, scale_xy, scale_wh
            )
        self.assertTrue(torch.allclose(bbox_res1, bbox_res2, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(bbox_res1, bbox_res3, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(score_res1, score_res2, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(score_res1, score_res3, rtol=1e-4, atol=1e-4))

        # test double
        bbox_res4, score_res4 = parallel_scale_back_batch(
            predicted_loc.clone().double(),
            predicted_score,
            dboxes_xywh,
            scale_xy,
            scale_wh,
        )
        self.assertEqual(bbox_res4, bbox_res2)
        self.assertEqual(score_res4, score_res2)
        self.assertTrue(bbox_res4.dtype == torch.float64)


class TestNMS(TestCase):
    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        """
        Python implementation of Encoder::decode_single, refer to \
            https://github.com/mlcommons/inference/blob/v0.7/others/cloud/single_stage_detector/pytorch/utils.py
        """
        # perform non-maximum suppression
        # Reference to https://github.com/amdegroot/ssd.pytorch

        bboxes_out = []
        scores_out = []
        labels_out = []
        for i, score in enumerate(scores_in.split(1, 1)):
            # skip background
            # print(score[score>0.90])
            if i == 0:
                continue
            score = score.squeeze(1)
            mask = score > 0.05
            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0:
                continue
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
            labels_out.extend([i] * len(candidates))
        bboxes_out, labels_out, scores_out = (
            torch.cat(bboxes_out, dim=0),
            torch.tensor(labels_out, dtype=torch.long),
            torch.cat(scores_out, dim=0),
        )
        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]

    def test_batch_nms_result(self):
        batch_size = 1
        number_boxes = 15130
        scale_xy = 0.1
        scale_wh = 0.2
        criteria = 0.50
        max_output = 200
        predicted_loc = torch.load(
            os.path.join(os.path.dirname(__file__), "data/nms_ploc.pt")
        )  # sizes: [1, 15130, 4]
        predicted_score = torch.load(
            os.path.join(os.path.dirname(__file__), "data/nms_plabel.pt")
        )  # sizes: [1, 15130, 81]
        dboxes_xywh = torch.load(
            os.path.join(os.path.dirname(__file__), "data/nms_dboxes_xywh.pt")
        )
        bboxes, probs = parallel_scale_back_batch(
            predicted_loc, predicted_score, dboxes_xywh, scale_xy, scale_wh
        )
        bboxes_clone = bboxes.clone()
        probs_clone = probs.clone()

        output = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(self.decode_single(bbox, prob, criteria, max_output))
        output2_raw = batch_score_nms(bboxes_clone, probs_clone, criteria, max_output)

        # test autocast
        with torch.cpu.amp.autocast():
            for datatype in (torch.bfloat16, torch.float32):
                bboxes_autocast = bboxes.clone().to(datatype)
                probs_autocast = probs.clone().to(datatype)
                output2_raw_autocast = batch_score_nms(
                    bboxes_autocast, probs_autocast, criteria, max_output
                )
                for i in range(3):
                    self.assertTrue(output2_raw_autocast[i].dtype == torch.float32)

        # Re-assembly the result
        output2 = []
        idx = 0
        for i in range(output2_raw[3].size(0)):
            output2.append(
                (
                    output2_raw[0][idx : idx + output2_raw[3][i]],
                    output2_raw[1][idx : idx + output2_raw[3][i]],
                    output2_raw[2][idx : idx + output2_raw[3][i]],
                )
            )
            idx += output2_raw[3][i]

        for i in range(batch_size):
            loc, label, prob = list(r for r in output[i])
            loc2, label2, prob2 = list(r for r in output2[i])
            self.assertTrue(torch.allclose(loc, loc2, rtol=1e-4, atol=1e-4))
            self.assertEqual(label, label2)
            self.assertTrue(torch.allclose(prob, prob2, rtol=1e-4, atol=1e-4))

        # test double
        output2_raw_double = batch_score_nms(
            bboxes.clone().double(), probs.clone().double(), criteria, max_output
        )
        self.assertEqual(output2_raw_double, output2_raw)
        self.assertTrue(output2_raw_double[0].dtype == torch.float64)

    def test_jit_trace_batch_nms(self):
        class Batch_NMS(nn.Module):
            def __init__(self, criteria, max_output):
                super(Batch_NMS, self).__init__()
                self.criteria = criteria
                self.max_output = max_output

            def forward(self, bboxes_clone, probs_clone):
                return batch_score_nms(
                    bboxes_clone, probs_clone, self.criteria, self.max_output
                )

        batch_size = 1
        number_boxes = 15130
        scale_xy = 0.1
        scale_wh = 0.2
        criteria = 0.50
        max_output = 200
        predicted_loc = torch.load(
            os.path.join(os.path.dirname(__file__), "data/nms_ploc.pt")
        )  # sizes: [1, 15130, 4]
        predicted_score = torch.load(
            os.path.join(os.path.dirname(__file__), "data/nms_plabel.pt")
        )  # sizes: [1, 15130, 81]
        dboxes_xywh = torch.load(
            os.path.join(os.path.dirname(__file__), "data/nms_dboxes_xywh.pt")
        )
        bboxes, probs = parallel_scale_back_batch(
            predicted_loc, predicted_score, dboxes_xywh, scale_xy, scale_wh
        )
        bboxes_clone = bboxes.clone()
        probs_clone = probs.clone()

        output = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(self.decode_single(bbox, prob, criteria, max_output))

        batch_score_nms_module = Batch_NMS(criteria, max_output)
        model_decode = torch.jit.trace(
            batch_score_nms_module, (bboxes_clone, probs_clone)
        )
        output2_raw = model_decode(bboxes_clone, probs_clone)

        # Re-assembly the result
        output2 = []
        idx = 0
        for i in range(output2_raw[3].size(0)):
            output2.append(
                (
                    output2_raw[0][idx : idx + output2_raw[3][i]],
                    output2_raw[1][idx : idx + output2_raw[3][i]],
                    output2_raw[2][idx : idx + output2_raw[3][i]],
                )
            )
            idx += output2_raw[3][i]

        for i in range(batch_size):
            loc, label, prob = list(r for r in output[i])
            loc2, label2, prob2 = list(r for r in output2[i])
            self.assertTrue(torch.allclose(loc, loc2, rtol=1e-4, atol=1e-4))
            self.assertEqual(label, label2)
            self.assertTrue(torch.allclose(prob, prob2, rtol=1e-4, atol=1e-4))

    def test_nms_kernel_result(self):
        batch_size = 1
        class_number = 81
        scale_xy = 0.1
        scale_wh = 0.2
        criteria = 0.50
        max_output = 200
        predicted_loc = torch.load(
            os.path.join(os.path.dirname(__file__), "data/nms_ploc.pt")
        )  # sizes: [1, 15130, 4]
        predicted_score = torch.load(
            os.path.join(os.path.dirname(__file__), "data/nms_plabel.pt")
        )  # sizes: [1, 15130, 81]
        dboxes_xywh = torch.load(
            os.path.join(os.path.dirname(__file__), "data/nms_dboxes_xywh.pt")
        )
        bboxes, probs = parallel_scale_back_batch(
            predicted_loc, predicted_score, dboxes_xywh, scale_xy, scale_wh
        )

        for bs in range(batch_size):
            loc = bboxes[bs].squeeze(0)
            for class_id in range(class_number):
                if class_id == 0:
                    # Skip the background
                    continue
                score = probs[bs, :, class_id]

                score_sorted, indices = torch.sort(score, descending=True)
                loc_sorted = torch.index_select(loc, 0, indices)

                result = nms(loc_sorted.clone(), score_sorted.clone(), criteria, True)
                result_ref = nms(loc.clone(), score.clone(), criteria, False)
                result_ref2 = nms(
                    loc_sorted.clone().to(dtype=torch.float64),
                    score_sorted.clone().to(dtype=torch.float64),
                    criteria,
                    True,
                )

                bbox_keep, _ = torch.sort(
                    torch.index_select(loc_sorted, 0, result).squeeze(0), 0
                )
                bbox_keep_ref, _ = torch.sort(
                    torch.index_select(loc, 0, result_ref).squeeze(0), 0
                )
                bbox_keep_ref2, _ = torch.sort(
                    torch.index_select(loc_sorted, 0, result_ref2).squeeze(0), 0
                )

                score_keep, _ = torch.sort(
                    torch.index_select(score_sorted, 0, result).squeeze(0), 0
                )
                score_keep_ref, _ = torch.sort(
                    torch.index_select(score, 0, result_ref).squeeze(0), 0
                )
                score_keep_ref2, _ = torch.sort(
                    torch.index_select(score_sorted, 0, result_ref2).squeeze(0), 0
                )

                self.assertEqual(result.size(0), result_ref.size(0))
                self.assertTrue(
                    torch.allclose(bbox_keep, bbox_keep_ref, rtol=1e-4, atol=1e-4)
                )
                self.assertTrue(
                    torch.allclose(score_keep, score_keep_ref, rtol=1e-4, atol=1e-4)
                )
                self.assertTrue(
                    torch.allclose(bbox_keep, bbox_keep_ref2, rtol=1e-4, atol=1e-4)
                )
                self.assertTrue(
                    torch.allclose(score_keep, score_keep_ref2, rtol=1e-4, atol=1e-4)
                )

                # test autocast
                with torch.cpu.amp.autocast():
                    result_autocast = nms(loc.clone(), score.clone(), criteria, False)
                    self.assertEqual(result_autocast, result_ref)

                # test double
                result_double = nms(
                    loc.clone().double(), score.clone().double(), criteria, False
                )
                self.assertEqual(result_double, result_ref)

    def test_rpn_nms_result(self):
        image_shapes = [(800, 824), (800, 1199)]
        min_size = 0
        nms_thresh = 0.7
        post_nms_top_n = 1000
        proposals = torch.load(
            os.path.join(os.path.dirname(__file__), "data/rpn_nms_proposals.pt")
        )
        objectness = torch.load(
            os.path.join(os.path.dirname(__file__), "data/rpn_nms_objectness.pt")
        )

        new_proposal = []
        new_score = []
        for proposal, score, im_shape in zip(
            proposals.clone(), objectness.clone(), image_shapes
        ):
            proposal[:, 0].clamp_(min=0, max=im_shape[0] - 1)
            proposal[:, 1].clamp_(min=0, max=im_shape[1] - 1)
            proposal[:, 2].clamp_(min=0, max=im_shape[0] - 1)
            proposal[:, 3].clamp_(min=0, max=im_shape[1] - 1)
            keep = (
                (
                    (proposal[:, 2] - proposal[:, 0] >= min_size)
                    & (proposal[:, 3] - proposal[:, 1] >= min_size)
                )
                .nonzero()
                .squeeze(1)
            )
            proposal = proposal[keep]
            score = score[keep]
            if nms_thresh > 0:
                keep = nms(proposal, score, nms_thresh)
                if post_nms_top_n > 0:
                    keep = keep[:post_nms_top_n]
            new_proposal.append(proposal[keep])
            new_score.append(score[keep])

        new_proposal_, new_score_ = rpn_nms(
            proposals, objectness, image_shapes, min_size, nms_thresh, post_nms_top_n
        )

        self.assertEqual(new_proposal, new_proposal_)
        self.assertEqual(new_score, new_score_)

        # test autocast
        with torch.cpu.amp.autocast():
            for datatype in (torch.bfloat16, torch.float32):
                proposals_autocast = proposals.clone().to(datatype)
                objectness_autocast = objectness.clone().to(datatype)
                new_proposal_autocast, new_score_autocast = rpn_nms(
                    proposals_autocast,
                    objectness_autocast,
                    image_shapes,
                    min_size,
                    nms_thresh,
                    post_nms_top_n,
                )
                self.assertTrue(new_proposal_autocast[0].dtype == torch.float32)
                self.assertTrue(new_score_autocast[0].dtype == torch.float32)

        # test double
        new_proposal_double, new_score_double = rpn_nms(
            proposals.clone().double(),
            objectness.clone().double(),
            image_shapes,
            min_size,
            nms_thresh,
            post_nms_top_n,
        )
        self.assertEqual(new_proposal_double, new_proposal)
        self.assertEqual(new_score_double, new_score)
        self.assertTrue(new_proposal_double[0].dtype == torch.float64)
        self.assertTrue(new_score_double[0].dtype == torch.float64)


if __name__ == "__main__":
    test = unittest.main()
