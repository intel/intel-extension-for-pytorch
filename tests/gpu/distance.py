import torch
import torch.nn as nn
import torch_ipex

dtype = torch.float
cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

pdist = nn.PairwiseDistance(p=2)
input1 = torch.randn(100, 128, device=cpu_device, dtype=dtype, requires_grad=True)
input2 = torch.randn(100, 128, device=cpu_device, dtype=dtype, requires_grad=True)
output = pdist(input1, input2)
print(output)
pdist_dpcpp = pdist.to("dpcpp")
input1_dpcpp = torch.randn(100, 128, device=dpcpp_device, dtype=dtype, requires_grad=True)
input2_dpcpp = torch.randn(100, 128, device=dpcpp_device, dtype=dtype, requires_grad=True)
output_dpcpp = pdist_dpcpp(input1, input2)
print(output_dpcpp.to(cpu_device))

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
input1 = torch.randn(100, 128, device=cpu_device, dtype=dtype, requires_grad=True)
input2 = torch.randn(100, 128, device=cpu_device, dtype=dtype, requires_grad=True)
output = cos(input1, input2)
print(output)
cos_dpcpp = cos.to("dpcpp")
input1_dpcpp = torch.randn(100, 128, device=dpcpp_device, dtype=dtype, requires_grad=True)
input2_dpcpp = torch.randn(100, 128, device=dpcpp_device, dtype=dtype, requires_grad=True)
output_dpcpp = cos_dpcpp(input1, input2)
print(output_dpcpp.to(cpu_device))
