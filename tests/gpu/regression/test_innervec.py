import torch
import intel_extension_for_pytorch
from torch.testing._internal.common_utils import TestCase
import pytest
import numpy as np
from numpy import random

cpu_device = torch.device('cpu')
xpu_device = torch.device('xpu')
torch.xpu.manual_seed(0)
random.seed(0)

# Thin wrapper around scalar types for convenient duck typing, bootstrapping
# off numpy's handling for real/complex scalars.
class ComplexScalar():
    def __init__(self, val):
        self.val = val
    def real(self): return np.real(self.val)
    def imag(self): return np.imag(self.val)

_COMPLEX_DIM = 0
class ComplexTensor():
    def __init__(self, re, im=None):
        # check dtypes
        if not isinstance(re, torch.Tensor):
            re = torch.Tensor(re)
        if im is None:
            im = torch.zeros_like(re)
        if not isinstance(im, torch.Tensor):
            im = torch.Tensor(im)
        if re.dtype != torch.float32 and re.dtype != torch.float64:
            re = re.type(torch.get_default_dtype())
        if im.dtype != torch.float32 and im.dtype != torch.float64:
            im = im.type(torch.get_default_dtype())
        if not re.shape == im.shape:
            raise RuntimeError(f'Shape of re {re.shape} must match im {im.shape}')
        self.tensor = torch.stack((re, im), dim=_COMPLEX_DIM)
        self.shape = re.shape

    def real(self):
        ind = [slice(None)]*len(self.tensor.shape)
        ind[_COMPLEX_DIM] = 0
        return self.tensor[tuple(ind)]
    def imag(self):
        ind = [slice(None)]*len(self.tensor.shape)
        ind[_COMPLEX_DIM] = 1
        return self.tensor[tuple(ind)]

    @property
    def requires_grad(self):
        return self.tensor.requires_grad
    @requires_grad.setter
    def requires_grad(self, val):
        self.tensor.requires_grad = val

    @property
    def device(self):
        return self.tensor.device
    @device.setter
    def device(self, val):
        self.tensor.device = val

    def numpy(self):
        real = self.real().numpy()
        imag = self.imag().numpy()
        return real + 1j*imag
    def detach(self):
        return ComplexTensor(self.real().detach(), self.imag().detach())
    def squeeze(self, *args, **kwargs):
        return ComplexTensor(self.real().squeeze(*args, **kwargs), self.imag().squeeze(*args, **kwargs))
    def to(self, *args, **kwargs):
        return ComplexTensor(self.real().to(*args, **kwargs), self.imag().to(*args, **kwargs))
    def cpu(self):
        return self.to('cpu')
    def cuda(self):
        return self.to('cuda')

    def float(self):
        return self.to(torch.float32)
    def double(self):
        return self.to(torch.float64)

    def conj(self):
        return ComplexTensor(self.real(), -self.imag())
    def normsq(self):
        return self.real()**2 + self.imag()**2
    def arg(self):
        return torch.atan2(self.imag(), self.real())

    def unsqueeze(self, dim):
        return ComplexTensor(self.real().unsqueeze(dim), self.imag().unsqueeze(dim))
    def permute(self, *dims):
        return ComplexTensor(self.real().permute(*dims), self.imag().permute(*dims))
    def reshape(self, *shape):
        return ComplexTensor(self.real().reshape(*shape), self.imag().reshape(*shape))

    @staticmethod
    def upcast(other):
        if isinstance(other, ComplexTensor):
            return other
        elif isinstance(other, torch.Tensor):
            return ComplexTensor(other)
        elif np.isscalar(other): # special case wrapping scalar for convenience
            return ComplexScalar(other)
        else:
            raise TypeError(f'Could upcast other of type {type(other)} for '
                            'binary op with ComplexTensor')

    def __add__(self, other):
        other = ComplexTensor.upcast(other)
        return ComplexTensor(self.real()+other.real(), self.imag()+other.imag())
    def __neg__(self):
        return ComplexTensor(-self.real(), -self.imag())
    def __sub__(self, other):
        other = ComplexTensor.upcast(other)
        return self + (-other)
    def __mul__(self, other):
        other = ComplexTensor.upcast(other)
        return ComplexTensor(self.real()*other.real() - self.imag()*other.imag(),
                             self.real()*other.imag() + self.imag()*other.real())
    def __truediv__(self, other):
        if np.isscalar(other):
            return self * (1/other)
        other = ComplexTensor.upcast(other)
        return self * (other**(-1))
    def __abs__(self):
        return torch.sqrt(self.normsq())
    def __pow__(self, b):
        norm = abs(self)
        arg = self.arg()
        # defines a choice of branch cut for non-integer b, matching numpy convention
        return ComplexTensor(torch.cos(b*arg) * norm**b, torch.sin(b*arg) * norm**b)

    def __matmul__(self, other):
        return ComplexTensor(
            self.real() @ other.real() - self.imag() @ other.imag(),
            self.real() @ other.imag() + self.imag() @ other.real())

    def __str__(self):
        return f'{self.real()} + 1j*{self.imag()}'
    def __repr__(self):
        return f'ComplexTensor({self.real()}, {self.imag()})'

    def __getitem__(self, ind):
        return ComplexTensor(self.real()[ind], self.imag()[ind])
    def __setitem__(self, ind, item):
        self.real()[ind] = item.real()
        self.imag()[ind] = item.imag()

    def _not_implemented(self, *args, **kwargs):
        raise NotImplementedError('Function is not implemented for ComplexTensor')
    __iadd__ = __imul__ = _not_implemented
    __lt__ = __le__ = __eq__ = __ne__ = __ge__ = __gt__ = _not_implemented
    __not__ = __and__ = __or__ = _not_implemented
    __rshift__ = __inv__ = __xor__ = _not_implemented
    __mod__ = __floordiv__ = _not_implemented
    __concat__ = _not_implemented

def inner_vector(v1, v2):
    """Complex inner product of (batched) vectors `v1` and `v2`"""
    assert v1.shape == v2.shape, 'vector shapes must match'
    inner = 0
    cpu_inner = 0

    cpu_v1 = v1.to(cpu_device)
    cpu_v2 = v2.to(cpu_device)
    for i in range(v1.shape[-1]):
        ###################################################################
        # HERE IS THE OP THAT PRODUCES WRONG RESLUT
        # 4-D complex mul (4D mul/add/sub)
        ###################################################################
        inner = v1[...,i]*v2[...,i].conj() + inner
        cpu_inner = cpu_v1[...,i]*cpu_v2[...,i].conj() + cpu_inner

        # WRONG
        if not (torch.allclose(cpu_inner.real(), inner.real().to(cpu_device)) and torch.allclose(cpu_inner.imag(), inner.imag().to(cpu_device))):
            diff_real = torch.max(torch.abs(cpu_inner.real() - inner.real().to(cpu_device)))
            diff_imag = torch.max(torch.abs(cpu_inner.imag() - inner.imag().to(cpu_device)))
            print (str(i) + ': MaxDiff - real: ' + str(diff_real))
            print (str(i) + ': MaxDiff - imag: ' + str(diff_imag))
            assert False
 
    return inner

class TestTorchMethod(TestCase):
    def test_inner_vector(self, dtype=torch.float):
        max_size = 50000
        for i in range(0,3):
            # set last dim size to 3-5
            lastdim = random.randint(3, 5)
            # 2D
            rand = torch.rand(max_size, lastdim, device=xpu_device, dtype=dtype)
            print (rand.shape)
            inner = inner_vector(ComplexTensor(rand, rand), ComplexTensor(rand, rand))
            # 3D
            dim1 = random.randint(1, int(max_size**0.5))
            dim2 = random.randint(1, int(max_size/dim1)+1)
            rand = torch.rand(dim1, dim2, lastdim, device=xpu_device, dtype=dtype)
            print (rand.shape)
            inner = inner_vector(ComplexTensor(rand, rand), ComplexTensor(rand, rand))
            # 4D
            dim1 = random.randint(1, int(max_size**0.33))
            dim2 = random.randint(1, int(max_size / dim1)+1)
            dim3 = random.randint(1, int(max_size / (dim1*dim2))+1)
            rand = torch.rand(dim1, dim2, dim3, lastdim, device=xpu_device, dtype=dtype)
            print (rand.shape)
            inner = inner_vector(ComplexTensor(rand, rand), ComplexTensor(rand, rand))
            # 5D
            dim1 = random.randint(1, int(max_size**0.25))
            dim2 = random.randint(1, int(max_size / dim1)+1)
            dim3 = random.randint(1, int(max_size / (dim1*dim2))+1)
            dim4 = random.randint(1, int(max_size / (dim1*dim2*dim3))+1)
            rand = torch.rand(dim1, dim2, dim3, dim4, lastdim, device=xpu_device, dtype=dtype)
            print (rand.shape)
            inner = inner_vector(ComplexTensor(rand, rand), ComplexTensor(rand, rand))
            # 6D
            dim1 = random.randint(1, int(max_size**0.16666))
            dim2 = random.randint(1, int(max_size / dim1)+1)
            dim3 = random.randint(1, int(max_size / (dim1*dim2))+1)
            dim4 = random.randint(1, int(max_size / (dim1*dim2*dim3))+1)
            dim5 = random.randint(1, int(max_size / (dim1*dim2*dim3*dim4))+1)
            rand = torch.rand(dim1, dim2, dim3, dim4, dim5, lastdim, device=xpu_device, dtype=dtype)
            print (rand.shape)
            inner = inner_vector(ComplexTensor(rand, rand), ComplexTensor(rand, rand))
