import torch
import warnings

def _numpy(x):
    if x.dtype==torch.bfloat16:
        warnings.warn("calling in ipex numpy which is not share mermory with torch tensor for bfloat16 input.")
        return torch._C._TensorBase.numpy(x.float())
    else:
        return torch._C._TensorBase.numpy(x)

torch.Tensor.numpy = _numpy

