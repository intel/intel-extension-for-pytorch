import torch
import intel_pytorch_extension as ipex
ipex.core.enable_auto_dnnl()


_torch_rand = torch.rand


def dpcpp_torch_rand(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_rand(*args, **kwargs)
    else:
        return _torch_rand(*args, **kwargs).to("dpcpp")


torch.rand = dpcpp_torch_rand

_torch_randint = torch.randint


def dpcpp_torch_randint(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_randint(*args, **kwargs)
    else:
        return _torch_randint(*args, **kwargs).to("dpcpp")


torch.randint = dpcpp_torch_randint

_torch_arange = torch.arange


def dpcpp_torch_arange(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_arange(*args, **kwargs)
    else:
        return _torch_arange(*args, **kwargs).to("dpcpp")


torch.arange = dpcpp_torch_arange

_torch_bartlett_window = torch.bartlett_window


def dpcpp_torch_bartlett_window(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_bartlett_window(*args, **kwargs)
    else:
        return _torch_bartlett_window(*args, **kwargs).to("dpcpp")


torch.bartlett_window = dpcpp_torch_bartlett_window

_torch_blackman_window = torch.blackman_window


def dpcpp_torch_blackman_window(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_blackman_window(*args, **kwargs)
    else:
        return _torch_blackman_window(*args, **kwargs).to("dpcpp")


torch.blackman_window = dpcpp_torch_blackman_window

_torch_empty = torch.empty


def dpcpp_torch_empty(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_empty(*args, **kwargs)
    else:
        return _torch_empty(*args, **kwargs).to("dpcpp")


torch.empty = dpcpp_torch_empty

_torch__empty_affine_quantized = torch._empty_affine_quantized


def dpcpp_torch__empty_affine_quantized(*args, **kwargs):
    if 'device' in kwargs:
        return _torch__empty_affine_quantized(*args, **kwargs)
    else:
        return _torch__empty_affine_quantized(*args, **kwargs).to("dpcpp")


torch._empty_affine_quantized = dpcpp_torch__empty_affine_quantized

_torch__empty_per_channel_affine_quantized = torch._empty_per_channel_affine_quantized


def dpcpp_torch__empty_per_channel_affine_quantized(*args, **kwargs):
    if 'device' in kwargs:
        return _torch__empty_per_channel_affine_quantized(*args, **kwargs)
    else:
        return _torch__empty_per_channel_affine_quantized(*args, **kwargs).to("dpcpp")


torch._empty_per_channel_affine_quantized = dpcpp_torch__empty_per_channel_affine_quantized

_torch_empty_strided = torch.empty_strided


def dpcpp_torch_empty_strided(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_empty_strided(*args, **kwargs)
    else:
        return _torch_empty_strided(*args, **kwargs).to("dpcpp")


torch.empty_strided = dpcpp_torch_empty_strided

_torch_eye = torch.eye


def dpcpp_torch_eye(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_eye(*args, **kwargs)
    else:
        return _torch_eye(*args, **kwargs).to("dpcpp")


torch.eye = dpcpp_torch_eye

_torch_full = torch.full


def dpcpp_torch_full(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_full(*args, **kwargs)
    else:
        return _torch_full(*args, **kwargs).to("dpcpp")


torch.full = dpcpp_torch_full

_torch_from_file = torch.from_file


def dpcpp_torch_from_file(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_from_file(*args, **kwargs)
    else:
        return _torch_from_file(*args, **kwargs).to("dpcpp")


torch.from_file = dpcpp_torch_from_file

_torch_from_numpy = torch.from_numpy


def dpcpp_torch_from_numpy(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_from_numpy(*args, **kwargs)
    else:
        return _torch_from_numpy(*args, **kwargs).to("dpcpp")


torch.from_numpy = dpcpp_torch_from_numpy

_torch_hann_window = torch.hann_window


def dpcpp_torch_hann_window(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_hann_window(*args, **kwargs)
    else:
        return _torch_hann_window(*args, **kwargs).to("dpcpp")


torch.hann_window = dpcpp_torch_hann_window

_torch_hamming_window = torch.hamming_window


def dpcpp_torch_hamming_window(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_hamming_window(*args, **kwargs)
    else:
        return _torch_hamming_window(*args, **kwargs).to("dpcpp")


torch.hamming_window = dpcpp_torch_hamming_window

_torch_linspace = torch.linspace


def dpcpp_torch_linspace(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_linspace(*args, **kwargs)
    else:
        return _torch_linspace(*args, **kwargs).to("dpcpp")


torch.linspace = dpcpp_torch_linspace

_torch_logspace = torch.logspace


def dpcpp_torch_logspace(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_logspace(*args, **kwargs)
    else:
        return _torch_logspace(*args, **kwargs).to("dpcpp")


torch.logspace = dpcpp_torch_logspace

_torch_ones = torch.ones


def dpcpp_torch_ones(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_ones(*args, **kwargs)
    else:
        return _torch_ones(*args, **kwargs).to("dpcpp")


torch.ones = dpcpp_torch_ones

_torch_scalar_tensor = torch.scalar_tensor


def dpcpp_torch_scalar_tensor(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_scalar_tensor(*args, **kwargs)
    else:
        return _torch_scalar_tensor(*args, **kwargs).to("dpcpp")


torch.scalar_tensor = dpcpp_torch_scalar_tensor

_torch_randn = torch.randn


def dpcpp_torch_randn(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_randn(*args, **kwargs)
    else:
        return _torch_randn(*args, **kwargs).to("dpcpp")


torch.randn = dpcpp_torch_randn

_torch_randperm = torch.randperm


def dpcpp_torch_randperm(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_randperm(*args, **kwargs)
    else:
        return _torch_randperm(*args, **kwargs).to("dpcpp")


torch.randperm = dpcpp_torch_randperm

_torch_range = torch.range


def dpcpp_torch_range(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_range(*args, **kwargs)
    else:
        return _torch_range(*args, **kwargs).to("dpcpp")


torch.range = dpcpp_torch_range

_torch_zeros = torch.zeros


def dpcpp_torch_zeros(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_zeros(*args, **kwargs)
    else:
        return _torch_zeros(*args, **kwargs).to("dpcpp")


torch.zeros = dpcpp_torch_zeros

_torch_sparse_coo_tensor = torch.sparse_coo_tensor


def dpcpp_torch_sparse_coo_tensor(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_sparse_coo_tensor(*args, **kwargs)
    else:
        return _torch_sparse_coo_tensor(*args, **kwargs).to("dpcpp")


torch.sparse_coo_tensor = dpcpp_torch_sparse_coo_tensor

_torch_tril_indices = torch.tril_indices


def dpcpp_torch_tril_indices(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_tril_indices(*args, **kwargs)
    else:
        return _torch_tril_indices(*args, **kwargs).to("dpcpp")


torch.tril_indices = dpcpp_torch_tril_indices

_torch_triu_indices = torch.triu_indices


def dpcpp_torch_triu_indices(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_triu_indices(*args, **kwargs)
    else:
        return _torch_triu_indices(*args, **kwargs).to("dpcpp")


torch.triu_indices = dpcpp_torch_triu_indices

_torch_normal = torch.normal


def dpcpp_torch_normal(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_normal(*args, **kwargs)
    else:
        return _torch_normal(*args, **kwargs).to("dpcpp")


torch.normal = dpcpp_torch_normal

_torch_tensor = torch.tensor


def dpcpp_torch_tensor(*args, **kwargs):
    if 'device' in kwargs:
        return _torch_tensor(*args, **kwargs)
    else:
        return _torch_tensor(*args, **kwargs).to("dpcpp")


torch.tensor = dpcpp_torch_tensor

