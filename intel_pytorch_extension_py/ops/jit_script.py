import torch
import intel_pytorch_extension as ipex
import _torch_ipex as core
from torch.jit._recursive import wrap_cpp_module

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

orig_script = torch.jit.script

def script_(obj, optimize=None, _frames_up=0, _rcb=None):
    torch.jit.script = orig_script
    jit_m = orig_script(obj, optimize=optimize, _frames_up=_frames_up+1, _rcb=_rcb)
    torch.jit.script = script_

    if core.get_jit_opt():
        # Disable mix precision in model fusion, since mixed precision cannot
        # bring any benefits for inference, but will lead to loss of accuracy
        orig_mixed_type = ipex.get_auto_mix_precision()
        ipex.enable_auto_mix_precision(None)
        jit_m = wrap_cpp_module(torch._C._jit_pass_fold_convbn(jit_m._c))
        ipex.enable_auto_mix_precision(orig_mixed_type)

    return jit_m


torch.jit.script = script_
