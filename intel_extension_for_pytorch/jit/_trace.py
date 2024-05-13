import torch
from functools import wraps

# For CPU, wrap torch.jit.trace to disable autocast cache when using torch.jit.trace
# within the scope of torch.cpu.amp.autocast.
# See https://github.com/pytorch/pytorch/pull/63552 for more information.

# For XPU, wrap torch.jit.trace to disable the check trace to avoid the double floating
# computing for the xpu platform which unsupports 2d block


def need_to_disable_check_trace_for_XPU(*args, **kwargs):
    device_type_list = []

    def check_input_tensor(arg):
        for elm in arg:
            if isinstance(elm, torch.Tensor):
                device_type_list.append(elm.device.type)
            else:
                check_input_tensor(elm)

    for arg in args:
        if isinstance(arg, torch.Tensor):
            device_type_list.append(arg.device.type)
        elif isinstance(arg, tuple) or isinstance(arg, list):
            check_input_tensor(arg)
        elif isinstance(arg, dict):
            check_input_tensor(list(arg.values()))
        else:
            pass

    if "example_inputs" in kwargs:
        example_inputs = kwargs["example_inputs"]
        if isinstance(example_inputs, torch.Tensor):
            device_type_list.append(example_inputs.device.type)
        elif isinstance(example_inputs, tuple) or isinstance(example_inputs, list):
            check_input_tensor(example_inputs)
        elif isinstance(example_inputs, dict):
            check_input_tensor(list(example_inputs.values()))
        else:
            pass

    is_xpu = all([elm == "xpu" for elm in device_type_list])
    if (
        is_xpu
        and ("check_trace" not in kwargs)
        and (not torch.xpu.has_2d_block_array())
    ):
        return True

    return False


def jit_trace_wrapper(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        prev = torch.is_autocast_cache_enabled()
        # For running CPU workload, disable autocast cache
        if torch.is_autocast_enabled("cpu"):
            torch.set_autocast_cache_enabled(False)

        # For running XPU workload and the platform unsupports 2d block,
        # the check_trace is here disabled in jit trace to avoid double computing
        if torch.xpu.is_available() and need_to_disable_check_trace_for_XPU(
            *args, **kwargs
        ):
            kwargs["check_trace"] = False

        traced = f(*args, **kwargs)
        torch.set_autocast_cache_enabled(prev)
        return traced

    return wrapper


torch.jit.trace = jit_trace_wrapper(torch.jit.trace)
