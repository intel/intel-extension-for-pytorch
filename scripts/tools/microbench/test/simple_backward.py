import torch
import microbench

microbench.enable_verbose()

# The following cmd will generate verbose
a = torch.randn(2)
a.requires_grad = True
b = torch.rand(2)
c = (a / b).sum()
c.backward()

microbench.disable_verbose()
