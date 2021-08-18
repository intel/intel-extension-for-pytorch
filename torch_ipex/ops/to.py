import torch
import torch_ipex._C as core

torch_to = torch.nn.Module.to

def apply(m, fn):
    for sub_module in m.children():
        apply(sub_module, fn)
    fn(m)
    return m

def to(module, *args, **kwargs):
    m = torch_to(module, *args, **kwargs)

    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

    if not device or device.type != "xpu":
        return m

    def mark_param(t):
        for param in t.parameters():
            core.set_parameter_tensor(param.data)

    return apply(m, mark_param)

torch.nn.Module.to = to
