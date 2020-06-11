import torch
import _torch_ipex as core
from torch.jit._recursive import wrap_cpp_module

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

orig_script = torch.jit.script

def script_(obj, optimize=None, _frames_up=0, _rcb=None):
    torch.jit.script = orig_script
    jit_m = orig_script(obj, optimize=optimize, _frames_up=_frames_up+1, _rcb=_rcb)
    torch.jit.script = script_

    if core.get_jit():
        # bypass buggy broadcastable ops in dnnl during folding
        core.disable_auto_dnnl()
        jit_m = wrap_cpp_module(torch._C._jit_pass_fold_convbn(jit_m._c))
        core.enable_auto_dnnl()

        jit_m = wrap_cpp_module(core._jit_prepack_conv_weight(jit_m._c))
    
    return jit_m


torch.jit.script = script_
