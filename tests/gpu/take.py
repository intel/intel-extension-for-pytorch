import torch
from torch import nn

import torch_ipex

src = torch.rand(2, 3)
print(src)

dst = torch.take(src, torch.tensor([0,2,5]))
print("dst = ", dst)

src_sycl = src.to("dpcpp");
idx_sycl = torch.tensor([0,2,5], device=torch.device("dpcpp"), dtype=torch.long)
print(idx_sycl.shape)
dst_sycl_1 = torch.take(src_sycl, idx_sycl)
# dst_sycl_2 = torch.take(dst_sycl_1, torch.tensor([0], device=torch.device("dpcpp"), dtype=torch.long))
print("dst_sycl_1 = ", dst_sycl_1.cpu())
# print("dst_sycl_2 = ", dst_sycl_2.cpu())

