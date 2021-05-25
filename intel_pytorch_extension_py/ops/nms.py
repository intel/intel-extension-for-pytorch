import torch

def nms(dets, scores, threshold, sorted=False):
    return torch.ops.torch_ipex.nms(dets, scores, threshold, sorted)
batch_score_nms = torch.ops.torch_ipex.batch_score_nms
parallel_scale_back_batch = torch.ops.torch_ipex.parallel_scale_back_batch
