import torch
import ipex
import microbench

microbench.enable_verbose()
a = torch.rand(2).cpu()
b = torch.rand(2).cpu()
c = a + b
microbench.disable_verbose()
