import torch
import intel_extension_for_pytorch  # noqa F401
from functools import wraps
from torch.nn.parallel.scatter_gather import _is_namedtuple


def override_tensor_totype():
    r"""Override _tensor_totype to avoid triggering fp64 error when printing XPU tensor on ATS-M"""

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

    torch._tensor_str.tensor_totype = fp64_tensor_totype_wrapper(
        torch._tensor_str.tensor_totype
    )


def override_assert_equal():
    r"""Override assertEqual to avoid triggering fp64 error on tensor comparison in test case"""

    def args_to_xpu(args):
        if torch.is_tensor(args) and args.is_xpu:
            return args.to("cpu")
        elif isinstance(args, (tuple, list)):
            args_list = list(args)
            for i, arg in enumerate(args_list):
                args_list[i] = args_to_xpu(arg)
            if isinstance(args, tuple):
                return tuple(args_list)
            else:
                return args_list
        else:
            return args

    def fp64_assert_equal_wrapper(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            args = args_to_xpu(args)
            return f(*args, **kwargs)

        return wrapper

    r"""
    In PyTorch design, `__allow_nonbracketed_mutation_flag` is a flag to forbid bare assignment
    to torch.backends.<cudnn|mkldnn>.enabled and friends when running test suite. This flag will
    be forced to set to False by function `disable_global_flags` which is defined in
    torch.testing._internal.common_utils when overriding TestCase.assertEqual. It may result in
    a runtime error on subsequent cudnn|mkldnn setting, if any. The function here is to override
    `disable_global_flags` with an empty one to keep the flag `__allow_nonbracketed_mutation_flag`
    from being changed.
    """

    def _disable_global_flags():
        pass

    torch.backends.disable_global_flags = _disable_global_flags
    from torch.testing._internal.common_utils import TestCase

    TestCase.assertEqual = fp64_assert_equal_wrapper(TestCase.assertEqual)


# background streams used for copying
_streams = None


def override_get_stream():
    r"""
    This function overrides `_get_stream` in PyTorch to provide XPU support.
    """

    def _get_stream(device: int):
        r"""
        Gets a background stream for copying between CPU and XPU.
        """
        global _streams
        if device == -1:
            return None
        if _streams is None:
            _streams = [None] * torch.xpu.device_count()
        if _streams[device] is None:
            _streams[device] = torch.xpu.Stream(device)
        return _streams[device]

    torch.nn.parallel._functions._get_stream = _get_stream
    return _get_stream


def override_recursive_to():
    r"""
    This function overrides `_recursive_to` in PyTorch to provide XPU support for data movement.
    """

    def _recursive_to(inputs, target_gpu, use_side_stream_for_tensor_copies):
        r"""
        Recursively moves input to the target_gpu, used in XPU distributed training.
        """

        def to_map(obj):
            if isinstance(obj, torch.Tensor):
                if obj.device == torch.device("xpu", target_gpu):
                    return (obj,)
                if not use_side_stream_for_tensor_copies:
                    return (obj.to(target_gpu),)
                else:
                    # Perform CPU -> GPU copies in a background stream. This code is
                    # motivated from similar logic in torch/nn/parallel/_functions.py
                    _get_stream = override_get_stream()
                    stream = _get_stream(target_gpu)
                    with torch.xpu.stream(stream):
                        output = obj.to(target_gpu)
                    # synchronize with the copy stream
                    with torch.xpu.device(target_gpu):
                        current_stream = torch.xpu.current_stream()
                        # Sync the current stream with the copy stream
                        current_stream.wait_stream(stream)
                        # nsure tensor memory is not reused until work on
                        # main stream is complete
                        output.record_stream(current_stream)
                    return (output,)
            if _is_namedtuple(obj):
                return [type(obj)(*args) for args in zip(*map(to_map, obj))]
            if isinstance(obj, tuple) and len(obj) > 0:
                return list(zip(*map(to_map, obj)))
            if isinstance(obj, list) and len(obj) > 0:
                return [list(i) for i in zip(*map(to_map, obj))]
            if isinstance(obj, dict) and len(obj) > 0:
                return [type(obj)(i) for i in zip(*map(to_map, obj.items()))]
            return [obj]

        # Avoid reference cycle
        try:
            res = to_map(inputs)
        finally:
            to_map = None  # type: ignore[assignment]
        return res

    torch.distributed.utils._recursive_to = _recursive_to


class WrapAPI:
    user_defined_src_dtype = None
    user_defined_dst_dtype = None
    only_device = False

    @classmethod
    def wrap_api_to(cls, api):
        @wraps(api)
        def new_api(*args, **kwargs):
            new_args = list(args)
            assert isinstance(
                args[0], torch.Tensor
            ), "torch.Tensor.to wrapper got non-Tensor for the 1st argument"
            src_dtype = args[0].dtype
            src_device = args[0].device
            dst_dtype = kwargs.get("dtype")
            dst_device = kwargs.get("device")
            if dst_dtype is None and dst_device is None:
                if len(args) > 1 and isinstance(
                    args[1], torch.Tensor
                ):  # torch.Tensor.to(other, ...)
                    kwargs["dtype"] = args[1].dtype
                    kwargs["device"] = args[1].device
                    new_args.pop(1)
                elif len(args) > 1 and isinstance(
                    args[1], torch.dtype
                ):  # torch.Tensor.to(dtype, ...)
                    kwargs["dtype"] = args[1]
                    kwargs["device"] = src_device
                    new_args.pop(1)
                elif len(args) > 2 and isinstance(
                    args[2], torch.dtype
                ):  # torch.Tensor.to(device, dtype, ...)
                    kwargs["dtype"] = args[2]
                    kwargs["device"] = args[1]
                    new_args.pop(2)
                    new_args.pop(1)
                elif len(args) > 1:  # torch.Tensor.to(device, ...)
                    kwargs["dtype"] = src_dtype
                    kwargs["device"] = args[1]
                    new_args.pop(1)
            elif dst_device is None:
                if len(args) > 1:  # torch.Tensor.to(device, dtype=dtype, ...)
                    kwargs["device"] = args[1]
                    new_args.pop(1)
                else:  # torch.Tensor.to(dtype=dtype, ...)
                    kwargs["device"] = src_device
            elif dst_dtype is None:  # torch.Tensor.to(device=device, ...)
                kwargs["dtype"] = src_dtype
            else:  # torch.Tensor.to(device=device, dtype=dtype, ...)
                pass

            new_args = tuple(new_args)
            if cls.only_device and "xpu" not in str(kwargs["device"]):
                pass
            elif kwargs["dtype"] == cls.user_defined_src_dtype:
                kwargs["dtype"] = cls.user_defined_dst_dtype
            return api(*new_args, **kwargs)

        return new_api

    @classmethod
    def wrap_api_create_size(cls, api):
        @wraps(api)
        def new_api(*args, **kwargs):
            new_args = list(args)
            dst_dtype = kwargs.get("dtype")
            dst_device = kwargs.get("device")
            if cls.only_device and "xpu" not in str(dst_device):
                return api(*args, **kwargs)
            if dst_dtype == cls.user_defined_src_dtype:
                kwargs["dtype"] = cls.user_defined_dst_dtype
            new_args = tuple(new_args)
            return api(*new_args, **kwargs)

        return new_api

    @classmethod
    def wrap_api_create_tensor(cls, api):
        @wraps(api)
        def new_api(*args, **kwargs):
            new_args = list(args)
            assert len(args) > 0 and isinstance(
                args[0], torch.Tensor
            ), f"Current api {api} got non-Tensor for the 1st arguement"
            dst_device = args[0].device
            dst_dtype = args[0].dtype
            resign_dtype = kwargs.get("dtype")
            resign_dev = kwargs.get("device")
            dst_device = resign_dev if resign_dev is not None else dst_device
            dst_dtype = resign_dtype if resign_dtype is not None else dst_dtype
            if cls.only_device and "xpu" not in str(dst_device):
                return api(*args, **kwargs)
            if dst_dtype == cls.user_defined_src_dtype:
                kwargs["dtype"] = cls.user_defined_dst_dtype
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
