import torch
import intel_extension_for_pytorch  # noqa
from functools import wraps

from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    get_overridable_functions,
    get_testing_overrides,
    is_tensor_method_or_property,
    TorchFunctionMode
)

__all__ = ["handle_torch_function", "has_torch_function", "get_overridable_functions",
           "get_testing_overrides", "is_tensor_method_or_property", "convert_default_dtype"]

import functools  # noqa
from functools import partial

# The dispatch table for tensor factory's __torch_function__ implementation.
HANDLED_FUNCTIONS_SUB = {}

DEFAULT_XPU_DEVICE = "xpu"
DEFAULT_DTYPE = torch.float


def implements_sub(torch_function):
    "Register a torch function override for SubTensor"
    HANDLED_FUNCTIONS_SUB[torch_function] = partial(torch_function, device=DEFAULT_XPU_DEVICE, dtype=DEFAULT_DTYPE)


implements_sub(torch.empty)


def set_default_tensor_type(tensor_type):
    class XPUDefaultTensorTypeMode(TorchFunctionMode):

        def __init__(self, tensor_type):
            if tensor_type is torch.xpu.FloatTensor:
                self.dtype = torch.float
            if tensor_type is torch.xpu.DoubleTensor:
                self.dtype = torch.float64

        def __torch_function__(self, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}

            if func in HANDLED_FUNCTIONS_SUB:
                return partial(func, device="xpu", dtype=self.dtype)(*args, **kwargs)

            return func(*args, **kwargs)

    if tensor_type in [torch.xpu.FloatTensor, torch.xpu.DoubleTensor]:
        mode_info = torch.overrides._TorchFunctionModeInfo()

        old = mode_info.get_mode()
        if old is None:
            inner = mode_info.base_mode_class(inner=None)
        else:
            inner = old

        mode = partial(XPUDefaultTensorTypeMode, tensor_type)(inner=inner)
        mode_info.set_mode(mode)


def enable_cl_to():
    class XPUDefaultTensorTypeMode(TorchFunctionMode):

        def __init__(self):
            pass

        def __torch_function__(self, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}

            if func in [torch.Tensor.to]:
                if "memory_format" in kwargs:
                    if kwargs["memory_format"] is torch.channels_last:
                        # This is very hacking code for PoC
                        if args[0].dim == 4:
                            # Error: Correct this for moving the 2nd dim to the last one
                            return args[0].transpose(1, -1).contiguous().transpose(1, -1)
                    if kwargs["memory_format"] is torch.channels_last_3d:
                        pass
            return func(*args, **kwargs)

    mode_info = torch.overrides._TorchFunctionModeInfo()

    old = mode_info.get_mode()
    if old is None:
        inner = mode_info.base_mode_class(inner=None)
    else:
        inner = old

    mode = partial(XPUDefaultTensorTypeMode)(inner=inner)
    mode_info.set_mode(mode)

def fp64_tensor_totype_wrapper(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        for arg in args:
            if torch.is_tensor(arg) and arg.is_xpu:
                return arg.to(torch.float)
        for k, kwarg in kwargs.items():
            if torch.is_tensor(kwarg) and kwarg.is_xpu:
                return kwarg.to(torch.float)
        return f(*args, **kwargs)
    return wrapper

def override_tensor_totype():
    r"""Override _tensor_totype to avoid triggering fp64 error when printing XPU tensor on ATS-M"""
    torch._tensor_str.tensor_totype = fp64_tensor_totype_wrapper(torch._tensor_str.tensor_totype)

def fp64_assert_equal_wrapper(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        args_list = list(args)
        for i, arg in enumerate(args_list):
            if torch.is_tensor(arg) and arg.is_xpu:
                args_list[i] = arg.to("cpu")
            elif isinstance(arg, (tuple, list)) and len(arg) > 0 and torch.is_tensor(arg[0]):
                tensors = list(arg)
                for j, tensor in enumerate(tensors):
                    if tensor.is_xpu:
                        tensors[j] = tensor.to("cpu")
                if isinstance(arg, (tuple)):
                    args_list[i] = tuple(tensors)
                else:
                    args_list[i] = tensor
        args = tuple(args_list)
        return f(*args, **kwargs)
    return wrapper

def override_assert_equal():
    r"""Override assertEqual to avoid triggering fp64 error on tensor comparison in test case"""
    override_disable_global_flags();
    from torch.testing._internal.common_utils import TestCase
    TestCase.assertEqual = fp64_assert_equal_wrapper(TestCase.assertEqual)

def _disable_global_flags():
    pass

def override_disable_global_flags():
    r"""
    In PyTorch design, `__allow_nonbracketed_mutation_flag` is a flag to forbid bare assignment 
    to torch.backends.<cudnn|mkldnn>.enabled and friends when running test suite. This flag will 
    be forced to set to False by function `disable_global_flags` which is defined in 
    torch.testing._internal.common_utils when overriding TestCase.assertEqual. It may result in 
    a runtime error on subsequent cudnn|mkldnn setting, if any. The function here is to override 
    `disable_global_flags` with an empty one to keep the flag `__allow_nonbracketed_mutation_flag` 
    from being changed.
    """
    torch.backends.disable_global_flags = _disable_global_flags

class WrapAPI:

    user_defined_src_dtype = None
    user_defined_dst_dtype = None
    only_device = False

    @classmethod
    def wrap_api_to(cls, api):
        @wraps(api)
        def new_api(*args, **kwargs):
            new_args = list(args)
            assert isinstance(args[0], torch.Tensor), "torch.Tensor.to wrapper got non-Tensor for the 1st argument"
            src_dtype = args[0].dtype
            src_device = args[0].device
            dst_dtype = kwargs.get("dtype")
            dst_device = kwargs.get("device")
            if dst_dtype is None and dst_device is None:
                if len(args) > 1 and isinstance(args[1], torch.Tensor):     # torch.Tensor.to(other, ...)
                    kwargs['dtype'] = args[1].dtype
                    kwargs['device'] = args[1].device
                    new_args.pop(1)
                elif len(args) > 1 and isinstance(args[1], torch.dtype):    # torch.Tensor.to(dtype, ...)
                    kwargs['dtype'] = args[1]
                    kwargs['device'] = src_device
                    new_args.pop(1)
                elif len(args) > 2 and isinstance(args[2], torch.dtype):    # torch.Tensor.to(device, dtype, ...)
                    kwargs['dtype'] = args[2]
                    kwargs['device'] = args[1]
                    new_args.pop(2)
                    new_args.pop(1)
                elif len(args) > 1:                                         # torch.Tensor.to(device, ...)
                    kwargs['dtype'] = src_dtype
                    kwargs['device'] = args[1]
                    new_args.pop(1)
            elif dst_device is None:
                if len(args) > 1:                                           # torch.Tensor.to(device, dtype=dtype, ...)
                    kwargs['device'] = args[1]
                    new_args.pop(1)
                else:                                                       # torch.Tensor.to(dtype=dtype, ...)
                    kwargs['device'] = src_device
            elif dst_dtype is None:                                         # torch.Tensor.to(device=device, ...)
                kwargs['dtype'] = src_dtype
            else:                                                           # torch.Tensor.to(device=device, dtype=dtype, ...)
                pass

            new_args = tuple(new_args)
            if cls.only_device and 'xpu' not in str(kwargs['device']):
                pass
            elif kwargs['dtype'] == cls.user_defined_src_dtype:
                kwargs['dtype'] = cls.user_defined_dst_dtype
            return api(*new_args, **kwargs)

        return new_api
    
    @classmethod
    def wrap_api_create_size(cls, api):
        @wraps(api)
        def new_api(*args, **kwargs):
            new_args = list(args)
            dst_dtype = kwargs.get("dtype")
            dst_device = kwargs.get("device")
            if cls.only_device and 'xpu' not in str(dst_device):
                return api(*args, **kwargs)
            if dst_dtype == cls.user_defined_src_dtype:
                kwargs['dtype'] = cls.user_defined_dst_dtype
            new_args = tuple(new_args)
            return api(*new_args, **kwargs)
        return new_api

    @classmethod
    def wrap_api_create_tensor(cls, api):
        @wraps(api)
        def new_api(*args, **kwargs):
            new_args = list(args)
            assert len(args) > 0 and isinstance(args[0], torch.Tensor), \
                f"Current api {api} got non-Tensor for the 1st arguement"
            dst_device = args[0].device
            dst_dtype = args[0].dtype
            resign_dtype = kwargs.get("dtype")
            resign_dev = kwargs.get("device")
            dst_device = resign_dev if resign_dev is not None else dst_device
            dst_dtype = resign_dtype if resign_dtype is not None else dst_dtype
            if cls.only_device and 'xpu' not in str(dst_device):
                return api(*args, **kwargs)
            if dst_dtype == cls.user_defined_src_dtype:
                kwargs['dtype'] = cls.user_defined_dst_dtype
            new_args = tuple(new_args)
            return api(*new_args, **kwargs)
        return new_api

def convert_default_dtype(src_dtype, dst_dtype, only_device=False):
    WrapAPI.user_defined_src_dtype = src_dtype
    WrapAPI.user_defined_dst_dtype = dst_dtype
    WrapAPI.only_device = only_device

    # hack to   
    # The apis implicitly included by torch.to include:
    torch.Tensor.to = WrapAPI.wrap_api_to(torch.Tensor.to)
    torch.Tensor.xpu = WrapAPI.wrap_api_to(torch.Tensor.xpu)

    # hack create size
    torch.tensor = WrapAPI.wrap_api_create_size(torch.tensor)
    torch.scalar_tensor = WrapAPI.wrap_api_create_size(torch.scalar_tensor)
    torch.empty_quantized = WrapAPI.wrap_api_create_size(torch.empty_quantized)
    torch.empty = WrapAPI.wrap_api_create_size(torch.empty)
    torch.ones = WrapAPI.wrap_api_create_size(torch.ones)
    torch.randint = WrapAPI.wrap_api_create_size(torch.randint)
    torch.zeros = WrapAPI.wrap_api_create_size(torch.zeros)
    torch.randn = WrapAPI.wrap_api_create_size(torch.randn)
    torch.rand = WrapAPI.wrap_api_create_size(torch.rand)
    torch.full = WrapAPI.wrap_api_create_size(torch.full)
    torch.arange = WrapAPI.wrap_api_create_size(torch.arange)
    torch.range = WrapAPI.wrap_api_create_size(torch.range)
    torch.logspace = WrapAPI.wrap_api_create_size(torch.logspace)
    torch.randperm = WrapAPI.wrap_api_create_size(torch.randperm)
    torch.linspace = WrapAPI.wrap_api_create_size(torch.linspace)
    torch.kaiser_window = WrapAPI.wrap_api_create_size(torch.kaiser_window)
    torch.hamming_window = WrapAPI.wrap_api_create_size(torch.hamming_window)
    torch.blackman_window = WrapAPI.wrap_api_create_size(torch.blackman_window)
    torch.hann_window = WrapAPI.wrap_api_create_size(torch.hann_window)
    torch.bartlett_window = WrapAPI.wrap_api_create_size(torch.bartlett_window)
    torch.tril_indices = WrapAPI.wrap_api_create_size(torch.tril_indices)
    torch.eye = WrapAPI.wrap_api_create_size(torch.eye)
    torch.empty_strided = WrapAPI.wrap_api_create_size(torch.empty_strided)
    torch.triu_indices = WrapAPI.wrap_api_create_size(torch.triu_indices)

    # hack create from other tensor
    torch.zeros_like = WrapAPI.wrap_api_create_tensor(torch.zeros_like)
    torch.ones_like = WrapAPI.wrap_api_create_tensor(torch.ones_like)
    torch.randn_like = WrapAPI.wrap_api_create_tensor(torch.randn_like)
    torch.rand_like = WrapAPI.wrap_api_create_tensor(torch.rand_like)
    torch.empty_like = WrapAPI.wrap_api_create_tensor(torch.empty_like)
    torch.full_like = WrapAPI.wrap_api_create_tensor(torch.full_like)
    torch.randint_like = WrapAPI.wrap_api_create_tensor(torch.randint_like)
    torch.asarray = WrapAPI.wrap_api_create_tensor(torch.asarray)
    torch.sparse_coo_tensor = WrapAPI.wrap_api_create_tensor(torch.sparse_coo_tensor)
