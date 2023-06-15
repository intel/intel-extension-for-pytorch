import torch
import intel_extension_for_pytorch as ipex

torch_function = ['rand', 'randint', 'arange', 'bartlett_window', 'blackman_window', \
                  'empty', '_empty_affine_quantized', '_empty_per_channel_affine_quantized', \
                  'empty_strided', 'eye', 'full', 'from_file', 'from_numpy', \
                  'hann_window', 'hamming_window', 'linspace', 'logspace', 'ones', \
                  'scalar_tensor', 'randn', 'randperm', 'range', 'zeros', \
                  'sparse_coo_tensor', 'tril_indices', 'triu_indices', 'normal', 'tensor']


def make_hooked_func(torch_func):
    def hooked_func(*args, **kwargs):
        if 'device' in kwargs:
            return torch_func(*args, **kwargs)
        else:
            return torch_func(*args, **kwargs).to(ipex.DEVICE)
    return hooked_func

for torch_func_name in torch_function:
    torch_fn = getattr(torch, torch_func_name)
    hooked_fn = make_hooked_func(torch_fn)
    setattr(torch, torch_func_name, hooked_fn)

