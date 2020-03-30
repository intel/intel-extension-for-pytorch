import torch
import torch_ipex

print("cpu isnan", torch.isnan(torch.tensor([1, float('nan'), 2])))
print("dpcpp isnan", torch.isnan(torch.tensor([1, float('nan'), 2], device=torch.device("dpcpp"))))
