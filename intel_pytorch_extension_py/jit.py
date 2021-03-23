import torch
import _torch_ipex as core

orig_trace = torch.jit.trace

def trace_(func, example_inputs, *args, **kwargs):
    '''
    # for autocast int path, need first run imperative path to cash the weight scales,
    # and then run the trace path to get unique grapg at differt time.

    #if core.is_autocast_enabled() and (torch.int8 == core.get_autocast_dtype()):
    if core.is_autocast_enabled():
        y = func(example_inputs)
    '''
    jit_m = orig_trace(func, example_inputs, *args, **kwargs)
    return jit_m

torch.jit.trace = trace_

