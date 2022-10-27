import torch

# This is a work-around to convert 3d tensor to channels last format.
# Theoretically, transpose(permute/view)-contiguous(to)-transpose(permute/view)
# can convert 3d tensor to channels last. However, this formula cannot convert all
# the shapes. It is because tensor suggest_memory_format may be different from
# exact memory format. is_contiguous() shows exact memory format and in c++ code,
# channels last chain is based on suggest_memory_format.
# We test several inputs, find that most of shapes can be converted to channels last,
# except for N1W format. It needs use as_strided to convert to channels last.
def tensor_to_channels_last_1d(t):
    assert t.dim() == 3

    if 1 == t.size(1):
        t = t.as_strided(t.size(), (t.size(1) * t.size(-1), 1, t.size(1)))
    else:
        t = t.view(t.size(0), t.size(1), 1, t.size(2))
        t = t.to(memory_format=torch.channels_last)
        t = t.view(t.size(0), t.size(1), t.size(3))
    return t

# Port from https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu/blob/920c7163c81d6c5098ba79ed482d57b1ded8521d/intel_extension_for_pytorch/xpu/utils.py#L6 and
def to_channels_last_1d(t):
    if isinstance(t, torch.nn.Module):
        for m in t.modules():
            for param in m.parameters():
                if isinstance(m, (torch.nn.Conv1d)):
                    if 3 == param.data.dim():
                        param.data = tensor_to_channels_last_1d(param.data)
        return t

    if 3 == t.dim():
        t = tensor_to_channels_last_1d(t)
    return t


# Port from https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu/blob/920c7163c81d6c5098ba79ed482d57b1ded8521d/intel_extension_for_pytorch/xpu/utils.py#L38
def is_contiguous_channels_last_1d(input):
    if 3 != input.dim():
        return False

    tmpTen = input.view(input.size(0), input.size(1), 1, input.size(2))
    if tmpTen.is_contiguous(memory_format=torch.channels_last):
        return True
    else:
        return False
