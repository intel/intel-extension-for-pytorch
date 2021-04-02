import torch
import _torch_ipex as core
from torch.jit._recursive import wrap_cpp_module

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

orig_script = torch.jit.script
orig_trace = torch.jit.trace

def script_(obj, optimize=None, _frames_up=0, _rcb=None):
    jit_m = orig_script(obj, optimize=optimize, _frames_up=_frames_up+1, _rcb=_rcb)
    if core.get_jit_opt() and hasattr(jit_m, '_c'):
        jit_m = wrap_cpp_module(torch._C._jit_pass_fold_convbn(jit_m._c))

    return jit_m

def trace_(func, example_inputs, *args, **kwargs):
    jit_m = orig_trace(func, example_inputs, *args, **kwargs)
    if core.get_jit_opt() and hasattr(jit_m, '_c'):
        jit_m = wrap_cpp_module(torch._C._jit_pass_fold_convbn(jit_m._c))

    return jit_m


torch.jit.script = script_
torch.jit.trace = trace_
