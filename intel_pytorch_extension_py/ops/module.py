import torch
import _torch_ipex as core


orig_module_to = torch.nn.Module.to

def module_to(self, *args, **kwargs):
    def prepack(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d):
            core.prepack_conv_weight(m.weight, m.padding, m.stride, m.dilation, m.groups)

    def prepack_reccur(m):
        prepack(m)
        for _, sub_m in m.named_children():
            prepack_reccur(sub_m)

    m = orig_module_to(self, *args, **kwargs)

    device = torch._C._nn._parse_to(*args, **kwargs)[0]
    if device and device.type == 'dpcpp':
        prepack_reccur(m)

    return m


torch.nn.Module.to = module_to