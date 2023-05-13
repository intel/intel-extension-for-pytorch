# coding: utf-8
import torch
import intel_extension_for_pytorch


# This is a WA. We will submit a PR to stock-PyTorch and make XPU backend
# supported in torch.Generator() API.
class Generator(torch._C.Generator):
    def __new__(cls, device=None):
        return intel_extension_for_pytorch._C.generator_new(device)
