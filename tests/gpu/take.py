import torch

import torch_ipex

src = torch.tensor([[4, 3, 5],[6, 7, 8]])
dst = torch.take(src, torch.tensor([0, 2, 5]))
print("dst = ", dst)
src_sycl = torch.tensor([[4, 3, 5],[6, 7, 8]], device = torch.device("dpcpp"))
dst_sycl = torch.take(src_sycl, torch.tensor([0,2,5]).to("dpcpp"))
print("dst_sycl = ", dst_sycl.to("cpu"))

