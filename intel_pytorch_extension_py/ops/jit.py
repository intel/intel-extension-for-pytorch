import torch
import _torch_ipex as core
from torch.jit._recursive import wrap_cpp_module

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

orig_script = torch.jit.script
orig_trace = torch.jit.trace

def script_(obj, optimize=None, _frames_up=0, _rcb=None):
    torch.jit.script = orig_script
    jit_m = orig_script(obj, optimize=optimize, _frames_up=_frames_up+1, _rcb=_rcb)
    torch.jit.script = script_

    mix_state = torch.bfloat16 if _C.get_mix_bf16_fp32() else torch.int8 if _C.get_mix_int8_fp32() else None
    # Disable mix precision in model fusion, since mixed precision cannot
    # bring any benefits for inference, but will lead to loss of accuracy
    _C.disable_mix_bf16_fp32()
    _C.disable_mix_int8_fp32()
    if _C.get_jit_opt() and hasattr(jit_m, '_c'):
        jit_m = wrap_cpp_module(torch._C._jit_pass_fold_convbn(jit_m._c))
    if mix_state == torch.bfloat16:
        _C.enable_mix_bf16_fp32()
    elif mix_state == torch.int8:
        _C.enable_mix_int8_fp32()
    return jit_m

def trace_(func, example_inputs, *args, **kwargs):
    # Disable mix precision. torch.jit.trace will check the traced output
    # against what is expected. Since mix precision will lead to
    # loss of accuracy, this will raise warning during torch.jit.trace
    mix_state = torch.bfloat16 if _C.get_mix_bf16_fp32() else torch.int8 if _C.get_mix_int8_fp32() else None
    _C.disable_mix_bf16_fp32()
    _C.disable_mix_int8_fp32()
    jit_m = orig_trace(func, example_inputs, *args, **kwargs)
    if _C.get_jit_opt() and hasattr(jit_m, '_c'):
        jit_m = wrap_cpp_module(torch._C._jit_pass_fold_convbn(jit_m._c))
    if mix_state == torch.bfloat16:
        _C.enable_mix_bf16_fp32()
    elif mix_state == torch.int8:
        _C.enable_mix_int8_fp32()
    return jit_m


torch.jit.script = script_
torch.jit.trace = trace_
