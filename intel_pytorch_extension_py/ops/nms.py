import torch

nms = torch.ops.torch_ipex.nms
batch_score_nms = torch.ops.torch_ipex.batch_score_nms
parallel_scale_back_batch = torch.ops.torch_ipex.parallel_scale_back_batch
